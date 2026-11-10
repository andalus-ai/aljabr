package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufPrepBytesTest {
    @Test
    void plansNoMinKQuantMatricesByExactPreparedBytes() {
        String previousQ2MaxBytes = System.getProperty("aljabr.gguf.q2k.cache_max_bytes");
        String previousQ4MaxBytes = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        String previousQ5MaxBytes = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q2k.cache_max_bytes", "320");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "288");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "288");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84 + 144 + 176);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeSimpleQ4KBlock(segment.asSlice(84, 144));
            writeQ5KBlock(segment.asSlice(228, 176), (byte) 0xFF, (byte) 0);
            GGUFTensorInfo q2 = new GGUFTensorInfo("q2.no_mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4 = new GGUFTensorInfo("q4.no_mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5 = new GGUFTensorInfo("q5.no_mins", new long[]{256, 1}, 13, 228, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q2, q4, q5), 0, segment, null);

            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertEquals(320L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q2));
            assertEquals(288L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q4));
            assertEquals(288L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q5));
            assertEquals(3, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertTrue(GgufTensorOps.shouldUsePreparedMatrixCache(model, q2, 1, 1, 320));
            assertTrue(GgufTensorOps.shouldUsePreparedMatrixCache(model, q4, 1, 1, 288));
            assertTrue(GgufTensorOps.shouldUsePreparedMatrixCache(model, q5, 1, 1, 288));
            assertFalse(GgufTensorOps.shouldUsePreparedMatrixCache(q4, 1, 1, 288));

            GgufTensorOps.PreparedMatrixCachePlan plan = GgufTensorOps.planPreparedMatrixCaches(model);
            assertEquals(3, plan.preparedCandidates());
            assertEquals(0, plan.skippedCacheTooSmallTensors());
            assertEquals(896L, plan.estimatedPreparedBytes());

            GgufTensorOps.PreparedMatrixCacheStats stats = GgufTensorOps.prepareMatrixCaches(model);
            assertTrue(stats.ready());
            assertEquals(3, stats.preparedTensors());
            assertEquals(896L, stats.cacheBytes());
            assertEquals(3, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertEquals(3, GgufTensorOps.clearPreparedMatrixCaches(model));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_max_bytes", previousQ2MaxBytes);
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previousQ4MaxBytes);
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previousQ5MaxBytes);
        }
    }
}
