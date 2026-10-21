package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.util.concurrent.atomic.AtomicIntegerArray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;

class GgufParRowsTest {
    @Test
    void forkJoinRowChunksCoverEveryRowExactlyOnce() {
        int rows = 11;
        int chunks = 4;
        AtomicIntegerArray visits = new AtomicIntegerArray(rows);

        GgufParallelConfig.runParallelRowChunks(rows, chunks, (start, end) -> {
            for (int row = start; row < end; row++) {
                visits.incrementAndGet(row);
            }
        });

        for (int row = 0; row < rows; row++) {
            assertEquals(1, visits.get(row));
        }
    }

    @Test
    void rowChunkHelpersShortCircuitSequentialRequests() {
        assertEquals(0, GgufRows.preparedRowChunks(false, 128, 4096));
        assertEquals(0, GgufRows.preparedRowChunks(true, 1, 4096));
        assertEquals(0, GgufRows.rawRowChunks(false, 128, 4096));
        assertEquals(0, GgufRows.rawRowChunks(true, 1, 4096));
    }

    @Test
    void preparedChunkHelperKeepsTinyRowsDirectEvenWhenParallelIsForced() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        try {
            assertEquals(0, GgufRows.preparedRowChunks(true, 4, 32));
            assertEquals(2, GgufRows.preparedRowChunks(true, 5, 32));
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
        }
    }

    @Test
    void rawChunkHelperSkipsWorkerChunksForConfinedSegments() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        try (Arena confined = Arena.ofConfined(); Arena shared = Arena.ofShared()) {
            assertEquals(0, GgufRows.rawRowChunks(confined.allocate(8), true, 4, 32));
            assertEquals(2, GgufRows.rawRowChunks(shared.allocate(8), true, 4, 32));
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
        }
    }

    @Test
    void rawChunkHelperCachesWorkerAccessDecisionPerSegment() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        GgufRows.clearRawWorkerAccessCache();
        try (Arena shared = Arena.ofShared()) {
            var segment = shared.allocate(8);
            assertEquals(0, GgufRows.rawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessFastCacheSize());

            assertEquals(2, GgufRows.rawRowChunks(segment, true, 4, 32));
            assertEquals(1, GgufRows.rawWorkerAccessCacheSize());
            assertEquals(1, GgufRows.recentRawWorkerAccessCacheSize());
            assertEquals(1, GgufRows.recentRawWorkerAccessFastCacheSize());

            assertEquals(2, GgufRows.rawRowChunks(segment, true, 4, 32));
            assertEquals(1, GgufRows.rawWorkerAccessCacheSize());
            assertEquals(1, GgufRows.recentRawWorkerAccessCacheSize());
            assertEquals(1, GgufRows.recentRawWorkerAccessFastCacheSize());
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
            GgufRows.clearRawWorkerAccessCache();
        }
    }

    @Test
    void rawChunkHelperSkipsWorkerAccessProbeForBelowThresholdWork() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1024");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        GgufRows.clearRawWorkerAccessCache();
        try (Arena shared = Arena.ofShared()) {
            assertEquals(0, GgufRows.rawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.rawRowChunks(shared.allocate(8), true, 2, 32));
            assertEquals(0, GgufRows.rawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessFastCacheSize());
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
            GgufRows.clearRawWorkerAccessCache();
        }
    }

    @Test
    void preparedMatVecParallelRowsCoverUnevenChunkSplits() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "2");
        GgufParallelConfig.resetParallelConfig();
        try {
            int rows = 5;
            int columns = 32;
            byte[] quants = new byte[rows * columns];
            float[] scales = new float[rows];
            for (int row = 0; row < rows; row++) {
                scales[row] = 1.0f;
                for (int column = 0; column < columns; column++) {
                    quants[row * columns + column] = (byte) (row + 1);
                }
            }
            GgufTensorOps.Q8Matrix matrix = new GgufTensorOps.Q8Matrix(columns, rows, 1, columns, quants, scales);

            float[] output = new float[rows];
            GgufTensorOps.matVecRows(matrix, ones(columns), output, rows, true);

            for (int row = 0; row < rows; row++) {
                assertEquals(columns * (row + 1), output[row], 0.0f);
            }
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
        }
    }
}
