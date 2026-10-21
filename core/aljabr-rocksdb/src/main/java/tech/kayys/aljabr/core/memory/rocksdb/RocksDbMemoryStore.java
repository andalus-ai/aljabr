package tech.kayys.aljabr.core.memory.rocksdb;

import org.rocksdb.BlockBasedTableConfig;
import org.rocksdb.BloomFilter;
import org.rocksdb.Cache;
import org.rocksdb.ClockCache;
import org.rocksdb.ColumnFamilyDescriptor;
import org.rocksdb.ColumnFamilyHandle;
import org.rocksdb.ColumnFamilyOptions;
import org.rocksdb.DBOptions;
import org.rocksdb.FlushOptions;
import org.rocksdb.LRUCache;
import org.rocksdb.Options;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;
import org.rocksdb.WriteOptions;

import tech.kayys.aljabr.core.memory.UnifiedMemoryStore;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * RocksDB-backed {@link UnifiedMemoryStore} with true zero-copy GET using the
 * Direct ByteBuffer API.
 *
 * <h2>Zero-copy GET</h2>
 * <p>
 * The RocksDB Java API (since 6.x) supports reading a value directly into a
 * caller-supplied {@code java.nio.ByteBuffer}. We allocate the output buffer
 * directly from the caller's {@link Arena} (via
 * {@code MemorySegment#asByteBuffer()})
 * so the result lands in arena-owned off-heap memory, with no {@code byte[]}
 * intermediate on the JVM heap at all.
 *
 * <h2>PUT</h2>
 * <p>
 * RocksDB's Java API does not yet expose a {@code ByteBuffer}-based put for
 * the default column family in all bindings. We therefore read the raw bytes
 * from the {@link MemorySegment} into a direct {@code ByteBuffer} (one
 * {@code memcopy}, no JVM heap allocation) and call the {@code ByteBuffer}
 * put overload.
 *
 * <h2>Performance options</h2>
 * <ul>
 * <li>Block cache: configurable via {@link Config#blockCacheMiB()}</li>
 * <li>Bloom filter: enabled by default (10 bits per key)</li>
 * <li>Async WAL disable: for write-heavy inference caches</li>
 * </ul>
 * @author bhangun
 */
public final class RocksDbMemoryStore implements UnifiedMemoryStore {

    private static final Logger LOG = Logger.getLogger(RocksDbMemoryStore.class.getName());

    private final RocksDB db;
    private final Options options;
    private final WriteOptions writeOptions;
    private final ReadOptions readOptions;
    private final Cache blockCache;

    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * Opens (or creates) a RocksDB database at {@code dbPath} with default
     * settings.
     *
     * @param dbPath directory path for the RocksDB files
     */
    public RocksDbMemoryStore(String dbPath) {
        this(dbPath, Config.defaults());
    }

    /**
     * Opens (or creates) a RocksDB database at {@code dbPath} with the supplied
     * config.
     *
     * @param dbPath directory path for the RocksDB files
     * @param config tuning configuration
     */
    public RocksDbMemoryStore(String dbPath, Config config) {
        RocksDB.loadLibrary();
        try {
            this.blockCache = new LRUCache(config.blockCacheMiB() * 1024L * 1024L);

            BlockBasedTableConfig tableConfig = new BlockBasedTableConfig()
                    .setBlockCache(blockCache)
                    .setFilterPolicy(new BloomFilter(10))
                    .setCacheIndexAndFilterBlocks(true);

            this.options = new Options()
                    .setCreateIfMissing(true)
                    .setTableFormatConfig(tableConfig)
                    .setMaxOpenFiles(config.maxOpenFiles())
                    .setWriteBufferSize(64L * 1024 * 1024) // 64 MiB memtable
                    .setMaxWriteBufferNumber(3)
                    .setTargetFileSizeBase(64L * 1024 * 1024);

            this.writeOptions = new WriteOptions()
                    .setDisableWAL(config.disableWal()) // disable WAL for ephemeral KV caches
                    .setSync(false);

            this.readOptions = new ReadOptions()
                    .setFillCache(true)
                    .setVerifyChecksums(false);

            this.db = RocksDB.open(options, dbPath);

            LOG.info(() -> "RocksDbMemoryStore opened at " + dbPath
                    + " [blockCache=" + config.blockCacheMiB() + " MiB"
                    + ", WAL=" + !config.disableWal() + "]");
        } catch (RocksDBException e) {
            throw new RuntimeException("Failed to initialize RocksDB at " + dbPath, e);
        }
    }

    // ── UnifiedMemoryStore ────────────────────────────────────────────────────

    /**
     * Writes the contents of {@code valueSegment} to RocksDB.
     *
     * <p>
     * The value is read from the off-heap segment into a thread-local direct
     * {@code ByteBuffer} (one native-to-native copy, no JVM heap array) and
     * then handed to RocksDB.
     *
     * @param key          the key bytes
     * @param valueSegment the off-heap value
     */
    @Override
    public void put(byte[] key, MemorySegment valueSegment) {
        long byteSize = valueSegment.byteSize();
        try {
            // Use the segment's DirectByteBuffer view directly — RocksDB reads from it.
            ByteBuffer keyBuf = ByteBuffer.allocateDirect(key.length);
            keyBuf.put(key).flip();

            // asByteBuffer() returns a view into the off-heap segment — no copy onto JVM
            // heap.
            ByteBuffer valBuf = valueSegment.asByteBuffer();

            db.put(writeOptions, keyBuf, valBuf);
        } catch (RocksDBException e) {
            throw new RuntimeException("RocksDB put failed for key (size=" + byteSize + ")", e);
        }
    }

    /**
     * Retrieves a value from RocksDB directly into the caller's {@link Arena}
     * memory.
     *
     * <p>
     * The read is <em>zero-copy on the Java heap</em>: RocksDB writes the value
     * bytes
     * into an off-heap {@link MemorySegment} allocated from {@code arena},
     * bypassing
     * any {@code byte[]} intermediary.
     *
     * @param key          the key to look up
     * @param expectedSize the expected byte size of the value (used for initial
     *                     allocation)
     * @param arena        the arena that will own the returned segment's memory
     * @return the value as a segment, or empty if the key was not found
     */
    @Override
    public Optional<MemorySegment> getZeroCopy(byte[] key, long expectedSize, Arena arena) {
        try {
            // Allocate from the caller's arena — this memory is off-heap,
            // owned by the arena, and lives until the arena is closed.
            MemorySegment outSeg = arena.allocate(expectedSize);
            ByteBuffer outBuf = outSeg.asByteBuffer(); // view into arena memory

            ByteBuffer keyBuf = ByteBuffer.allocateDirect(key.length);
            keyBuf.put(key).flip();

            // RocksDB writes directly into outBuf (which is backed by the arena segment).
            int bytesRead = db.get(readOptions, keyBuf, outBuf);

            if (bytesRead == RocksDB.NOT_FOUND || bytesRead < 0) {
                return Optional.empty();
            }

            // If the value was larger than expectedSize, we need to retry with a bigger
            // buffer.
            if (bytesRead > expectedSize) {
                MemorySegment bigSeg = arena.allocate(bytesRead);
                ByteBuffer bigBuf = bigSeg.asByteBuffer();
                keyBuf.rewind();
                db.get(readOptions, keyBuf, bigBuf);
                return Optional.of(bigSeg.asSlice(0, bytesRead));
            }

            return Optional.of(outSeg.asSlice(0, bytesRead));
        } catch (RocksDBException e) {
            throw new RuntimeException("RocksDB get failed", e);
        }
    }

    @Override
    public void delete(byte[] key) {
        try {
            db.delete(writeOptions, key);
        } catch (RocksDBException e) {
            throw new RuntimeException("RocksDB delete failed", e);
        }
    }

    @Override
    public void flush() {
        try (FlushOptions fo = new FlushOptions().setWaitForFlush(true)) {
            db.flush(fo);
        } catch (RocksDBException e) {
            throw new RuntimeException("RocksDB flush failed", e);
        }
    }

    @Override
    public void close() {
        readOptions.close();
        writeOptions.close();
        if (db != null)
            db.close();
        if (options != null)
            options.close();
        if (blockCache != null)
            blockCache.close();
    }

    // ── Configuration record ──────────────────────────────────────────────────

    /**
     * Immutable tuning configuration for {@link RocksDbMemoryStore}.
     */
    public record Config(
            long blockCacheMiB,
            int maxOpenFiles,
            boolean disableWal) {
        public static Config defaults() {
            return new Config(256L, 512, false);
        }

        /**
         * Optimised for ephemeral inference KV caches: large cache, no WAL overhead.
         */
        public static Config forKvCache(long cacheMiB) {
            return new Config(cacheMiB, 256, true);
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder {
            private long blockCacheMiB = 256L;
            private int maxOpenFiles = 512;
            private boolean disableWal = false;

            public Builder blockCacheMiB(long mib) {
                this.blockCacheMiB = mib;
                return this;
            }

            public Builder maxOpenFiles(int n) {
                this.maxOpenFiles = n;
                return this;
            }

            public Builder disableWal(boolean v) {
                this.disableWal = v;
                return this;
            }

            public Config build() {
                return new Config(blockCacheMiB, maxOpenFiles, disableWal);
            }
        }
    }
}
