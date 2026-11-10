package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;

/**
 * Shared validation and row scheduling for matrix execution.
 *
 * <p>The public prepared-matrix overloads all enforce the same vector, output,
 * and row-count contracts. Raw FFM-backed paths also need the same parallel
 * fallback when a confined {@code MemorySegment} cannot be read from worker
 * threads, so both policies live here instead of being repeated per format.</p>
 * @author bhangun
 */
final class GgufRows {
    private static final int DIRECT_PREPARED_ROW_LIMIT = 4;
    private static final int RECENT_RAW_ACCESS_SLOTS = 256;
    private static final int RECENT_RAW_ACCESS_MASK = RECENT_RAW_ACCESS_SLOTS - 1;
    private static final Thread RAW_WORKER_ACCESS_PROBE = new Thread("aljabr-gguf-raw-worker-access-probe");
    private static final ThreadLocal<RecentRawWorkerAccesses> RECENT_RAW_WORKER_ACCESSES =
            ThreadLocal.withInitial(RecentRawWorkerAccesses::new);
    private static final Map<MemorySegment, Boolean> RAW_WORKER_ACCESS_CACHE = new WeakHashMap<>();

    private GgufRows() {
    }

    static int checkPreparedMatVec(int columns, int rows, float[] vector, float[] output, int rowCount) {
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        if (vector.length < columns) {
            throw new IllegalArgumentException(
                    "Vector length " + vector.length + " is smaller than columns " + columns);
        }
        if (rowCount < 0 || rowCount > rows) {
            throw new IllegalArgumentException(
                    "Requested row count " + rowCount + " is outside tensor rows " + rows);
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                    "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }
        return columns;
    }

    static void runPreparedRows(boolean parallel, int rowCount, int columns, RowRangeTask task) {
        if (!parallel || rowCount <= 1 || shouldRunPreparedRowsDirect(rowCount)) {
            task.run(0, rowCount);
            return;
        }
        int chunks = preparedRowChunks(parallel, rowCount, columns);
        if (chunks > 0) {
            GgufParallelConfig.runParallelRowChunks(rowCount, chunks, task);
            return;
        }
        task.run(0, rowCount);
    }

    static int preparedRowChunks(boolean parallel, int rowCount, int columns) {
        if (!parallel || rowCount <= 1 || shouldRunPreparedRowsDirect(rowCount)) {
            return 0;
        }
        return GgufParallelConfig.parallelChunkCount(parallel, rowCount, columns);
    }

    static boolean shouldRunPreparedRowsDirect(int rowCount) {
        return rowCount > 1 && rowCount <= DIRECT_PREPARED_ROW_LIMIT;
    }

    static int rawRowChunks(boolean parallel, int rowCount, int columns) {
        if (!parallel || rowCount <= 1) {
            return 0;
        }
        return GgufParallelConfig.parallelChunkCount(parallel, rowCount, columns);
    }

    static int rawRowChunks(MemorySegment data, boolean parallel, int rowCount, int columns) {
        if (!parallel || rowCount <= 1) {
            return 0;
        }
        int chunks = GgufParallelConfig.parallelChunkCount(parallel, rowCount, columns);
        if (chunks <= 0 || !rawWorkerAccessible(data)) {
            return 0;
        }
        return chunks;
    }

    static boolean rawWorkerAccessible(MemorySegment data) {
        RecentRawWorkerAccesses recent = RECENT_RAW_WORKER_ACCESSES.get();
        Boolean recentAccessible = recent.get(data);
        if (recentAccessible != null) {
            return recentAccessible;
        }
        synchronized (RAW_WORKER_ACCESS_CACHE) {
            Boolean cached = RAW_WORKER_ACCESS_CACHE.get(data);
            if (cached != null) {
                recent.put(data, cached);
                return cached;
            }
        }
        boolean accessible = data.isAccessibleBy(RAW_WORKER_ACCESS_PROBE);
        synchronized (RAW_WORKER_ACCESS_CACHE) {
            Boolean cached = RAW_WORKER_ACCESS_CACHE.putIfAbsent(data, accessible);
            boolean resolved = cached == null ? accessible : cached;
            recent.put(data, resolved);
            return resolved;
        }
    }

    static int rawWorkerAccessCacheSize() {
        synchronized (RAW_WORKER_ACCESS_CACHE) {
            return RAW_WORKER_ACCESS_CACHE.size();
        }
    }

    static void clearRawWorkerAccessCache() {
        synchronized (RAW_WORKER_ACCESS_CACHE) {
            RAW_WORKER_ACCESS_CACHE.clear();
        }
        RECENT_RAW_WORKER_ACCESSES.get().clear();
    }

    static int recentRawWorkerAccessCacheSize() {
        return RECENT_RAW_WORKER_ACCESSES.get().size();
    }

    static int recentRawWorkerAccessFastCacheSize() {
        return RECENT_RAW_WORKER_ACCESSES.get().fastSize();
    }

    static void runRawParallelRows(int rowCount, int chunks, RowRangeTask task) {
        GgufParallelConfig.runParallelRowChunks(rowCount, chunks, task);
    }

    static void runRawRows(boolean parallel, int rowCount, int columns, RowRangeTask task) {
        if (!parallel || rowCount <= 1) {
            task.run(0, rowCount);
            return;
        }
        int chunks = rawRowChunks(parallel, rowCount, columns);
        if (chunks > 0) {
            try {
                runRawParallelRows(rowCount, chunks, task);
                return;
            } catch (WrongThreadException ignored) {
                // Confined FFM segments cannot be read by worker threads; keep correctness and fall back.
            }
        }
        task.run(0, rowCount);
    }

    private static final class RecentRawWorkerAccesses {
        private final RecentRawWorkerAccess[] accesses = new RecentRawWorkerAccess[RECENT_RAW_ACCESS_SLOTS];
        private RecentRawWorkerAccess last;

        private Boolean get(MemorySegment segment) {
            Boolean accessible = accessibleIfMatches(last, segment);
            if (accessible != null) {
                return accessible;
            }
            int slot = slot(segment);
            RecentRawWorkerAccess recent = accesses[slot];
            accessible = accessibleIfMatches(recent, segment);
            if (accessible != null) {
                last = recent;
                return accessible;
            }
            if (recent != null && recentExpired(recent)) {
                accesses[slot] = null;
            }
            return null;
        }

        private void put(MemorySegment segment, boolean accessible) {
            RecentRawWorkerAccess recent = new RecentRawWorkerAccess(new WeakReference<>(segment), accessible);
            last = recent;
            accesses[slot(segment)] = recent;
        }

        private int size() {
            int size = 0;
            for (int index = 0; index < accesses.length; index++) {
                RecentRawWorkerAccess recent = accesses[index];
                if (recent == null) {
                    continue;
                }
                if (recent.segment().get() == null) {
                    accesses[index] = null;
                } else {
                    size++;
                }
            }
            return size;
        }

        private int fastSize() {
            if (last == null) {
                return 0;
            }
            if (recentExpired(last)) {
                last = null;
                return 0;
            }
            return 1;
        }

        private void clear() {
            last = null;
            for (int index = 0; index < accesses.length; index++) {
                accesses[index] = null;
            }
        }

        private static Boolean accessibleIfMatches(RecentRawWorkerAccess recent, MemorySegment segment) {
            if (recent == null) {
                return null;
            }
            MemorySegment cachedSegment = recent.segment().get();
            return cachedSegment == segment ? recent.accessible() : null;
        }

        private static boolean recentExpired(RecentRawWorkerAccess recent) {
            return recent.segment().get() == null;
        }

        private static int slot(MemorySegment segment) {
            return System.identityHashCode(segment) & RECENT_RAW_ACCESS_MASK;
        }
    }

    private record RecentRawWorkerAccess(
            WeakReference<MemorySegment> segment,
            boolean accessible) {
    }
}
