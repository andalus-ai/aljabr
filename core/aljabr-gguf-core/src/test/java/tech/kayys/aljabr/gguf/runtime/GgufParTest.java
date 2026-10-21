package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;

class GgufParTest {
    @Test
    void adaptsParallelMatVecToWorkSizeAndConfiguredChunks() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1024");
        System.setProperty("aljabr.gguf.parallel_threads", "3");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "2");
        GgufParallelConfig.resetParallelConfig();
        try {
            assertFalse(GgufTensorOps.shouldParallelize(false, 1024, 1024));
            assertFalse(GgufTensorOps.shouldParallelize(true, 1, 1024));
            assertFalse(GgufTensorOps.shouldParallelize(true, 2, 256));
            assertTrue(GgufTensorOps.shouldParallelize(true, 4, 256));
            assertEquals(6, GgufTensorOps.parallelChunkCount(64));
            assertEquals(4, GgufTensorOps.parallelChunkCount(4));
            assertEquals(1, GgufTensorOps.parallelChunkCount(1));
            assertEquals(0, GgufTensorOps.parallelChunkCount(false, 1024, 1024));
            assertEquals(0, GgufTensorOps.parallelChunkCount(true, 1, 1024));
            assertEquals(0, GgufTensorOps.parallelChunkCount(true, 2, 256));
            assertEquals(4, GgufTensorOps.parallelChunkCount(true, 4, 256));
            assertEquals(6, GgufTensorOps.parallelChunkCount(true, 64, 1024));
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
        }
    }

    @Test
    void usesTwoParallelChunksPerThreadByDefault() {
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_threads", "3");
        System.clearProperty("aljabr.gguf.parallel_chunks_per_thread");
        GgufParallelConfig.resetParallelConfig();
        try {
            assertEquals(6, GgufTensorOps.parallelChunkCount(64));
            assertEquals(4, GgufTensorOps.parallelChunkCount(4));
            assertEquals(1, GgufTensorOps.parallelChunkCount(1));
        } finally {
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
        }
    }

    @Test
    void reusesParallelConfigUntilExplicitReset() {
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        try {
            assertEquals(2, GgufTensorOps.parallelChunkCount(64));

            System.setProperty("aljabr.gguf.parallel_threads", "4");
            assertEquals(2, GgufTensorOps.parallelChunkCount(64));

            GgufParallelConfig.resetParallelConfig();
            assertEquals(4, GgufTensorOps.parallelChunkCount(64));
        } finally {
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
        }
    }

    @Test
    void cachesRequestedParallelChunkCountsUntilConfigReset() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        try {
            assertEquals(0, GgufParallelConfig.recentChunkCacheSize());
            assertEquals(0, GgufParallelConfig.recentChunkFastCacheSize());

            assertEquals(2, GgufParallelConfig.parallelChunkCount(true, 4, 32));
            assertEquals(1, GgufParallelConfig.recentChunkCacheSize());
            assertEquals(1, GgufParallelConfig.recentChunkFastCacheSize());

            assertEquals(2, GgufParallelConfig.parallelChunkCount(true, 4, 32));
            assertEquals(1, GgufParallelConfig.recentChunkCacheSize());
            assertEquals(1, GgufParallelConfig.recentChunkFastCacheSize());

            System.setProperty("aljabr.gguf.parallel_threads", "4");
            assertEquals(2, GgufParallelConfig.parallelChunkCount(true, 4, 32));

            GgufParallelConfig.resetParallelConfig();
            assertEquals(0, GgufParallelConfig.recentChunkCacheSize());
            assertEquals(0, GgufParallelConfig.recentChunkFastCacheSize());
            assertEquals(4, GgufParallelConfig.parallelChunkCount(true, 4, 32));
            assertEquals(1, GgufParallelConfig.recentChunkCacheSize());
            assertEquals(1, GgufParallelConfig.recentChunkFastCacheSize());
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
        }
    }
}
