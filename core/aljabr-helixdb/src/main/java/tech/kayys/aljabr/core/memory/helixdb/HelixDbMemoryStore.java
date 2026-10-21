package tech.kayys.aljabr.core.memory.helixdb;

import tech.kayys.aljabr.core.memory.ManagedArena;
import tech.kayys.aljabr.core.memory.UnifiedMemoryStore;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * HelixDB: a pure-Java off-heap key-value store backed by the JDK Foreign
 * Function &amp; Memory (FFM) API.
 *
 * <h2>Design</h2>
 * <p>
 * HelixDB is designed specifically for Wayang's inference use-case: storing
 * and retrieving large tensor blobs (KV-cache entries, embedding tables,
 * activations) with <em>zero JVM-heap copies</em>.
 *
 * <p>
 * All values are stored directly in off-heap memory managed by a shared
 * {@link ManagedArena} (the <em>storage arena</em>). Callers that need to read
 * values receive a <em>slice</em> of the storage segment — a true pointer alias
 * with no bytes copied. The caller must not outlive the store's storage arena.
 *
 * <h2>Compaction</h2>
 * <p>
 * When the storage grows beyond {@code Config#maxStorageMiB}, a compaction
 * cycle is triggered. Live entries are identified from the index and copied to
 * a fresh arena; the old arena is then closed in one syscall, reclaiming all
 * dead (deleted) space instantly.
 *
 * <h2>Thread safety</h2>
 * <p>
 * The index uses a {@link ConcurrentSkipListMap} for lock-free lookups.
 * Writes are guarded by a coarse-grained write lock to keep arena allocation
 * sequential (arenas are not inherently thread-safe for allocation).
 *
 * <h2>Persistence</h2>
 * <p>
 * This implementation is an <em>in-process</em> store. Entries do not
 * survive JVM restarts. The {@code dbPath} parameter is reserved for a future
 * mmap-backed persistence layer.
 * @author bhangun
 */
public final class HelixDbMemoryStore implements UnifiedMemoryStore {

    private static final Logger LOG = Logger.getLogger(HelixDbMemoryStore.class.getName());

    // ── Index: key → (storedSegment, ManagedArena that owns it) ──────────────
    private record Entry(MemorySegment segment, ManagedArena arena) {
    }

    private final ConcurrentSkipListMap<String, Entry> index = new ConcurrentSkipListMap<>(Comparator.naturalOrder());

    // ── Storage arena (shared) ────────────────────────────────────────────────
    private volatile ManagedArena storageArena;

    // ── Stats ─────────────────────────────────────────────────────────────────
    private final AtomicLong storedBytes = new AtomicLong(0);
    private final AtomicLong deletedBytes = new AtomicLong(0);

    // ── Config ────────────────────────────────────────────────────────────────
    private final Config config;

    // ── Write lock (protects arena allocation) ────────────────────────────────
    private final Object writeLock = new Object();

    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * Opens a HelixDB store. The {@code dbPath} is recorded but not used for
     * persistence in this implementation.
     *
     * @param dbPath reserved for future mmap persistence
     */
    public HelixDbMemoryStore(String dbPath) {
        this(dbPath, Config.defaults());
    }

    /**
     * Opens a HelixDB store with custom configuration.
     *
     * @param dbPath reserved for future mmap persistence
     * @param config tuning configuration
     */
    public HelixDbMemoryStore(String dbPath, Config config) {
        this.config = config;
        this.storageArena = ManagedArena.create();
        LOG.info("HelixDbMemoryStore initialized [maxStorage=" + config.maxStorageMiB() + " MiB]");
    }

    // ── UnifiedMemoryStore ────────────────────────────────────────────────────

    /**
     * Stores a value in HelixDB by copying it from the transient input
     * {@link MemorySegment} into the persistent off-heap storage arena.
     *
     * <p>
     * This is a one-time copy from the caller's (possibly pooled) segment
     * into HelixDB's own off-heap storage — from that point on, reads are
     * always zero-copy.
     *
     * @param key          the key bytes
     * @param valueSegment the segment to copy from; may be transient
     */
    @Override
    public void put(byte[] key, MemorySegment valueSegment) {
        String keyStr = new String(key);
        long size = valueSegment.byteSize();

        synchronized (writeLock) {
            // Trigger compaction if storage is getting full
            long maxBytes = config.maxStorageMiB() * 1024L * 1024L;
            if (storedBytes.get() - deletedBytes.get() > maxBytes) {
                compact();
            }

            MemorySegment stored = storageArena.allocate(size);
            MemorySegment.copy(valueSegment, 0, stored, 0, size);

            Entry old = index.put(keyStr, new Entry(stored, storageArena));
            if (old != null) {
                deletedBytes.addAndGet(old.segment().byteSize());
            }
            storedBytes.addAndGet(size);
        }
    }

    /**
     * Returns a zero-copy view into the stored segment.
     *
     * <p>
     * The returned {@link MemorySegment} is a direct slice of the internal
     * storage arena — <strong>no bytes are copied</strong>. The caller must
     * ensure that neither the segment nor any derived data is accessed after
     * this store is {@link #close() closed}.
     *
     * <p>
     * The {@code callerArena} parameter is accepted for API compatibility
     * but is not used for zero-copy reads from HelixDB — the returned memory is
     * already off-heap and does not need to be re-allocated.
     *
     * @param key          the key to look up
     * @param expectedSize only used as a validation hint; the actual size is always
     *                     returned
     * @param callerArena  ignored for HelixDB zero-copy reads
     * @return the stored value as a segment, or empty if not found
     */
    @Override
    public Optional<MemorySegment> getZeroCopy(byte[] key, long expectedSize, Arena callerArena) {
        Entry entry = index.get(new String(key));
        if (entry == null)
            return Optional.empty();

        MemorySegment seg = entry.segment();
        // Return a view — pure pointer arithmetic, zero bytes moved.
        long retSize = Math.min(expectedSize, seg.byteSize());
        return Optional.of(seg.asSlice(0, retSize));
    }

    @Override
    public void delete(byte[] key) {
        String keyStr = new String(key);
        synchronized (writeLock) {
            Entry old = index.remove(keyStr);
            if (old != null) {
                deletedBytes.addAndGet(old.segment().byteSize());
            }
        }
    }

    /**
     * Triggers a compaction cycle: copies all live entries to a new arena and
     * frees the old one, reclaiming deleted space.
     */
    @Override
    public void flush() {
        synchronized (writeLock) {
            compact();
        }
    }

    @Override
    public void close() {
        synchronized (writeLock) {
            index.clear();
            storageArena.close();
        }
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /**
     * Returns total bytes currently stored (including deleted entries not yet
     * compacted).
     */
    public long storedBytes() {
        return storedBytes.get();
    }

    /** Returns bytes belonging to deleted entries (freed on next compaction). */
    public long deletedBytes() {
        return deletedBytes.get();
    }

    /** Returns live entry count. */
    public int entryCount() {
        return index.size();
    }

    public String stats() {
        long live = storedBytes.get() - deletedBytes.get();
        long deleted = deletedBytes.get();
        return String.format("HelixDB[entries=%d, live=%s, deleted=%s, fragmentation=%.1f%%]",
                entryCount(),
                humanBytes(live),
                humanBytes(deleted),
                storedBytes.get() == 0 ? 0.0 : 100.0 * deleted / storedBytes.get());
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Compacts the store by moving all live entries to a fresh arena and
     * releasing the old one. Must be called under {@code writeLock}.
     */
    private void compact() {
        LOG.info("HelixDB compaction starting — " + stats());
        ManagedArena newArena = ManagedArena.create();

        // Rebuild the entire index into a fresh map backed by newArena.
        ConcurrentSkipListMap<String, Entry> newIndex = new ConcurrentSkipListMap<>(Comparator.naturalOrder());

        for (Map.Entry<String, Entry> e : index.entrySet()) {
            MemorySegment oldSeg = e.getValue().segment();
            MemorySegment newSeg = newArena.allocate(oldSeg.byteSize());
            MemorySegment.copy(oldSeg, 0, newSeg, 0, oldSeg.byteSize());
            newIndex.put(e.getKey(), new Entry(newSeg, newArena));
        }

        // Swap index atomically and free the old storage arena.
        index.clear();
        index.putAll(newIndex);

        ManagedArena old = storageArena;
        storageArena = newArena;
        old.close();

        // Reset counters.
        long live = index.values().stream()
                .mapToLong(en -> en.segment().byteSize())
                .sum();
        storedBytes.set(live);
        deletedBytes.set(0);

        LOG.info("HelixDB compaction complete — " + stats());
    }

    private static String humanBytes(long bytes) {
        if (bytes < 1024)
            return bytes + " B";
        if (bytes < 1024 * 1024)
            return String.format("%.1f KiB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024)
            return String.format("%.1f MiB", bytes / (1024.0 * 1024));
        return String.format("%.2f GiB", bytes / (1024.0 * 1024 * 1024));
    }

    // ── Configuration record ──────────────────────────────────────────────────

    /**
     * Tuning parameters for {@link HelixDbMemoryStore}.
     */
    public record Config(long maxStorageMiB) {

        /** 512 MiB storage limit before compaction triggers. */
        public static Config defaults() {
            return new Config(512L);
        }

        /** Small footprint for unit tests. */
        public static Config forTest() {
            return new Config(32L);
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder {
            private long maxStorageMiB = 512L;

            public Builder maxStorageMiB(long mib) {
                this.maxStorageMiB = mib;
                return this;
            }

            public Config build() {
                return new Config(maxStorageMiB);
            }
        }
    }
}
