package tech.kayys.aljabr.gguf.runtime;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * Row-chunk scheduling helpers for Java-native GGUF matrix work.
 *
 * <p>Parallel thresholds are intentionally centralized here so each kernel can
 * ask the same policy before paying stream parallelism overhead.</p>
 * @author bhangun
 */
final class GgufParallelConfig {
    private static final long DEFAULT_PARALLEL_MIN_OPS = 131_072L;
    private static final int DEFAULT_PARALLEL_CHUNKS_PER_THREAD = 2;
    private static final int RECENT_CHUNK_SLOTS = 256;
    private static final int RECENT_CHUNK_MASK = RECENT_CHUNK_SLOTS - 1;
    private static final ThreadLocal<RecentChunkCounts> RECENT_CHUNK_COUNTS =
            ThreadLocal.withInitial(RecentChunkCounts::new);
    private static volatile ParallelConfig cachedParallelConfig;

    private GgufParallelConfig() {
    }

    static boolean shouldParallelize(boolean requested, int rowCount, int columns) {
        if (!requested || rowCount <= 1) {
            return false;
        }
        return (long) rowCount * Math.max(0, columns) >= parallelConfig().minOps();
    }

    static int parallelChunkCount(int rowCount) {
        if (rowCount <= 1) {
            return 1;
        }
        ParallelConfig config = parallelConfig();
        long chunks = Math.min((long) rowCount, (long) config.threads() * config.chunksPerThread());
        return (int) Math.max(1L, chunks);
    }

    static int parallelChunkCount(boolean requested, int rowCount, int columns) {
        if (!requested || rowCount <= 1) {
            return 0;
        }
        ParallelConfig config = parallelConfig();
        RecentChunkCounts recent = RECENT_CHUNK_COUNTS.get();
        Integer cached = recent.get(config, rowCount, columns);
        if (cached != null) {
            return cached;
        }
        int chunks = parallelChunkCount(config, rowCount, columns);
        recent.put(config, rowCount, columns, chunks);
        return chunks;
    }

    private static int parallelChunkCount(ParallelConfig config, int rowCount, int columns) {
        if ((long) rowCount * Math.max(0, columns) < config.minOps()) {
            return 0;
        }
        long chunks = Math.min((long) rowCount, (long) config.threads() * config.chunksPerThread());
        return chunks <= 1L ? 0 : (int) chunks;
    }

    static void runParallelRowChunks(int rowCount, int chunks, RowRangeTask task) {
        if (chunks <= 1) {
            task.run(0, rowCount);
            return;
        }
        int rowsPerChunk = rowCount / chunks;
        int extraRows = rowCount % chunks;
        ForkJoinPool.commonPool()
                .invoke(new RowChunkAction(rowsPerChunk, extraRows, 0, chunks, task));
    }

    static void resetParallelConfig() {
        cachedParallelConfig = null;
        RECENT_CHUNK_COUNTS.get().clear();
    }

    static int recentChunkCacheSize() {
        return RECENT_CHUNK_COUNTS.get().size();
    }

    static int recentChunkFastCacheSize() {
        return RECENT_CHUNK_COUNTS.get().fastSize();
    }

    private static ParallelConfig parallelConfig() {
        ParallelConfig config = cachedParallelConfig;
        if (config != null) {
            return config;
        }
        String minOpsProperty = System.getProperty("aljabr.gguf.parallel_min_ops");
        String threadsProperty = System.getProperty("aljabr.gguf.parallel_threads");
        String chunksProperty = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        ParallelConfig updated = new ParallelConfig(
                Math.max(0L, parseLongProperty(minOpsProperty, DEFAULT_PARALLEL_MIN_OPS)),
                Math.max(1, parseIntProperty(threadsProperty, Runtime.getRuntime().availableProcessors())),
                Math.max(1, parseIntProperty(chunksProperty, DEFAULT_PARALLEL_CHUNKS_PER_THREAD)));
        cachedParallelConfig = updated;
        return updated;
    }

    private static long parseLongProperty(String configured, long defaultValue) {
        if (configured == null) {
            return defaultValue;
        }
        try {
            return Long.decode(configured.trim());
        } catch (RuntimeException ignored) {
            return defaultValue;
        }
    }

    private static int parseIntProperty(String configured, int defaultValue) {
        if (configured == null) {
            return defaultValue;
        }
        try {
            return Integer.decode(configured.trim());
        } catch (RuntimeException ignored) {
            return defaultValue;
        }
    }

    private record ParallelConfig(
            long minOps,
            int threads,
            int chunksPerThread) {
    }

    private static final class RecentChunkCounts {
        private final RecentChunkCount[] counts = new RecentChunkCount[RECENT_CHUNK_SLOTS];
        private RecentChunkCount last;

        private Integer get(ParallelConfig config, int rowCount, int columns) {
            Integer chunks = chunksIfMatches(last, config, rowCount, columns);
            if (chunks != null) {
                return chunks;
            }
            int slot = slot(config, rowCount, columns);
            RecentChunkCount count = counts[slot];
            chunks = chunksIfMatches(count, config, rowCount, columns);
            if (chunks != null) {
                last = count;
            }
            return chunks;
        }

        private void put(ParallelConfig config, int rowCount, int columns, int chunks) {
            RecentChunkCount count = new RecentChunkCount(config, rowCount, columns, chunks);
            last = count;
            counts[slot(config, rowCount, columns)] = count;
        }

        private int size() {
            int size = 0;
            for (RecentChunkCount count : counts) {
                if (count != null) {
                    size++;
                }
            }
            return size;
        }

        private int fastSize() {
            return last == null ? 0 : 1;
        }

        private void clear() {
            last = null;
            for (int index = 0; index < counts.length; index++) {
                counts[index] = null;
            }
        }

        private static Integer chunksIfMatches(
                RecentChunkCount count,
                ParallelConfig config,
                int rowCount,
                int columns) {
            return count != null
                    && count.config() == config
                    && count.rowCount() == rowCount
                    && count.columns() == columns
                    ? count.chunks()
                    : null;
        }

        private static int slot(ParallelConfig config, int rowCount, int columns) {
            int hash = System.identityHashCode(config);
            hash = 31 * hash + rowCount;
            hash = 31 * hash + columns;
            return hash & RECENT_CHUNK_MASK;
        }
    }

    private record RecentChunkCount(
            ParallelConfig config,
            int rowCount,
            int columns,
            int chunks) {
    }

    private static final class RowChunkAction extends RecursiveAction {
        private final int rowsPerChunk;
        private final int extraRows;
        private final int startChunk;
        private final int endChunk;
        private final RowRangeTask task;

        private RowChunkAction(
                int rowsPerChunk,
                int extraRows,
                int startChunk,
                int endChunk,
                RowRangeTask task) {
            this.rowsPerChunk = rowsPerChunk;
            this.extraRows = extraRows;
            this.startChunk = startChunk;
            this.endChunk = endChunk;
            this.task = task;
        }

        @Override
        protected void compute() {
            int chunkCount = endChunk - startChunk;
            if (chunkCount <= 1) {
                runChunk(startChunk);
                return;
            }
            int midChunk = startChunk + (chunkCount >>> 1);
            invokeAll(
                    new RowChunkAction(rowsPerChunk, extraRows, startChunk, midChunk, task),
                    new RowChunkAction(rowsPerChunk, extraRows, midChunk, endChunk, task));
        }

        private void runChunk(int chunk) {
            int start = chunk * rowsPerChunk + Math.min(chunk, extraRows);
            int end = start + rowsPerChunk + (chunk < extraRows ? 1 : 0);
            if (start < end) {
                task.run(start, end);
            }
        }
    }
}

/**
 * Callback used by {@link GgufParallelConfig} to process a half-open row range.
 */
@FunctionalInterface
interface RowRangeTask {
    void run(int startRow, int endRow);
}
