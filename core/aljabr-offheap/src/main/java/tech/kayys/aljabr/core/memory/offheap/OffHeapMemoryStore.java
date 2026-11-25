package tech.kayys.aljabr.core.memory.offheap;

import tech.kayys.aljabr.core.memory.BorrowableMemoryStore;
import tech.kayys.aljabr.core.memory.MemoryStoreLease;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.StampedLock;
import java.util.logging.Logger;

/**
 * In-process off-heap key-value store backed by the JDK FFM API.
 *
 * <p>Values are copied into a shared storage arena on {@link #put(byte[],
 * MemorySegment)}. {@link #borrow(byte[])} then exposes a direct alias without
 * copying. The returned lease pins the current storage generation: compaction,
 * deletion/replacement and shutdown wait until outstanding leases are closed.
 *
 * <p>This implementation is not persistent; values do not survive JVM restarts.
 */
public final class OffHeapMemoryStore implements BorrowableMemoryStore {

    private static final Logger LOG = Logger.getLogger(OffHeapMemoryStore.class.getName());
    private record Entry(MemorySegment segment, Arena arena) {}

    private final ConcurrentSkipListMap<String, Entry> index =
            new ConcurrentSkipListMap<>(Comparator.naturalOrder());
    private final StampedLock lifecycleLock = new StampedLock();
    private final AtomicLong storedBytes = new AtomicLong();
    private final AtomicLong deletedBytes = new AtomicLong();
    private final Config config;
    private Arena storageArena;
    private boolean closed;

    public OffHeapMemoryStore(String dbPath) {
        this(dbPath, Config.defaults());
    }

    public OffHeapMemoryStore(String dbPath, Config config) {
        this.config = config;
        this.storageArena = Arena.ofShared();
        LOG.info("In-process off-heap store initialized [maxStorage=" + config.maxStorageMiB() + " MiB]");
    }

    @Override
    public void put(byte[] key, MemorySegment valueSegment) {
        long stamp = lifecycleLock.writeLock();
        try {
            ensureOpen();
            String keyString = keyString(key);
            long size = valueSegment.byteSize();
            long maxBytes = config.maxStorageMiB() * 1024L * 1024L;
            if (storedBytes.get() - deletedBytes.get() > maxBytes) {
                compact();
            }
            MemorySegment stored = storageArena.allocate(size);
            MemorySegment.copy(valueSegment, 0, stored, 0, size);
            Entry old = index.put(keyString, new Entry(stored, storageArena));
            if (old != null) deletedBytes.addAndGet(old.segment().byteSize());
            storedBytes.addAndGet(size);
        } finally {
            lifecycleLock.unlockWrite(stamp);
        }
    }

    @Override
    public Optional<MemoryStoreLease> borrow(byte[] key) {
        long stamp = lifecycleLock.readLock();
        boolean transferred = false;
        try {
            ensureOpen();
            Entry entry = index.get(keyString(key));
            if (entry == null) return Optional.empty();
            MemoryStoreLease lease = MemoryStoreLease.create(
                    entry.segment(), () -> lifecycleLock.unlockRead(stamp));
            transferred = true;
            return Optional.of(lease);
        } finally {
            if (!transferred) lifecycleLock.unlockRead(stamp);
        }
    }

    /**
     * Legacy unleased view. Unsafe across concurrent compaction or close.
     * Prefer {@link #borrow(byte[])} and close the lease after the final access.
     */
    @Deprecated(forRemoval = true)
    @Override
    public Optional<MemorySegment> getZeroCopy(byte[] key, long expectedSize, Arena callerArena) {
        long stamp = lifecycleLock.readLock();
        try {
            ensureOpen();
            Entry entry = index.get(keyString(key));
            if (entry == null) return Optional.empty();
            if (expectedSize < 0) throw new IllegalArgumentException("expectedSize must be non-negative");
            long size = expectedSize == 0 ? entry.segment().byteSize()
                    : Math.min(expectedSize, entry.segment().byteSize());
            return Optional.of(entry.segment().asSlice(0, size));
        } finally {
            lifecycleLock.unlockRead(stamp);
        }
    }

    @Override
    public void delete(byte[] key) {
        long stamp = lifecycleLock.writeLock();
        try {
            ensureOpen();
            Entry old = index.remove(keyString(key));
            if (old != null) deletedBytes.addAndGet(old.segment().byteSize());
        } finally {
            lifecycleLock.unlockWrite(stamp);
        }
    }

    @Override
    public void flush() {
        long stamp = lifecycleLock.writeLock();
        try {
            ensureOpen();
            compact();
        } finally {
            lifecycleLock.unlockWrite(stamp);
        }
    }

    @Override
    public void close() {
        long stamp = lifecycleLock.writeLock();
        try {
            if (closed) return;
            closed = true;
            index.clear();
            storageArena.close();
        } finally {
            lifecycleLock.unlockWrite(stamp);
        }
    }

    public long storedBytes() { return storedBytes.get(); }
    public long deletedBytes() { return deletedBytes.get(); }
    public int entryCount() { return index.size(); }

    public String stats() {
        long live = storedBytes.get() - deletedBytes.get();
        long deleted = deletedBytes.get();
        return String.format("OffHeapMemoryStore[entries=%d, live=%s, deleted=%s, fragmentation=%.1f%%]",
                entryCount(), humanBytes(live), humanBytes(deleted),
                storedBytes.get() == 0 ? 0.0 : 100.0 * deleted / storedBytes.get());
    }

    private void compact() {
        LOG.info("Off-heap store compaction starting — " + stats());
        Arena newArena = Arena.ofShared();
        try {
            ConcurrentSkipListMap<String, Entry> newIndex =
                    new ConcurrentSkipListMap<>(Comparator.naturalOrder());
            for (Map.Entry<String, Entry> entry : index.entrySet()) {
                MemorySegment oldSegment = entry.getValue().segment();
                MemorySegment newSegment = newArena.allocate(oldSegment.byteSize());
                MemorySegment.copy(oldSegment, 0, newSegment, 0, oldSegment.byteSize());
                newIndex.put(entry.getKey(), new Entry(newSegment, newArena));
            }
            Arena oldArena = storageArena;
            index.clear();
            index.putAll(newIndex);
            storageArena = newArena;
            newArena = null;
            oldArena.close();
            long live = index.values().stream().mapToLong(e -> e.segment().byteSize()).sum();
            storedBytes.set(live);
            deletedBytes.set(0);
        } finally {
            if (newArena != null) newArena.close();
        }
        LOG.info("Off-heap store compaction complete — " + stats());
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("Memory store is closed");
    }

    private static String keyString(byte[] key) {
        return new String(key, StandardCharsets.UTF_8);
    }

    private static String humanBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KiB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MiB", bytes / (1024.0 * 1024));
        return String.format("%.2f GiB", bytes / (1024.0 * 1024 * 1024));
    }

    public record Config(long maxStorageMiB) {
        public static Config defaults() { return new Config(512L); }
        public static Config forTest() { return new Config(32L); }
        public static Builder builder() { return new Builder(); }
        public static final class Builder {
            private long maxStorageMiB = 512L;
            public Builder maxStorageMiB(long mib) { this.maxStorageMiB = mib; return this; }
            public Config build() { return new Config(maxStorageMiB); }
        }
    }
}
