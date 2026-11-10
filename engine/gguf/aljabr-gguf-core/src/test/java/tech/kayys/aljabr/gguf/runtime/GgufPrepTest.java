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

class GgufPrepTest {
    @Test
    void plansAndPreparesOnlyMatricesThatFitSharedCacheBucket() {
        String previous = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.PreparedMatrixCachePlan plan = GgufTensorOps.planPreparedMatrixCaches(model);
            assertEquals(2, plan.matrixTensors());
            assertEquals(1, plan.preparedCandidates());
            assertEquals(1, plan.skippedCacheTooSmallTensors());
            assertEquals(320L, plan.estimatedPreparedBytes());

            GgufTensorOps.PreparedMatrixCacheStats stats = GgufTensorOps.prepareMatrixCaches(model);
            assertTrue(stats.ready());
            assertEquals(1, stats.preparedCandidates());
            assertEquals(1, stats.preparedTensors());
            assertEquals(1, stats.skippedCacheTooSmallTensors());
            assertEquals(1, stats.cacheEntries());
            assertEquals(320L, stats.cacheBytes());
            assertEquals(1, GgufTensorOps.clearPreparedMatrixCaches(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previous);
        }
    }

    @Test
    void preparesSupportedMatrixCachesForWholeModel() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(196);
            GGUFModel model = GgufPrepFx.mixedPreparedModel(segment);

            GgufTensorOps.PreparedMatrixCacheStats stats = GgufTensorOps.prepareMatrixCaches(model);

            assertTrue(GgufTensorOps.supportsPreparedMatVecType(12));
            assertFalse(GgufTensorOps.supportsPreparedMatVecType(0));
            assertTrue(stats.ready());
            assertEquals(5, stats.scannedTensors());
            assertEquals(4, stats.matrixTensors());
            assertEquals(3, stats.preparedCandidates());
            assertEquals(3, stats.preparedTensors());
            assertEquals(1, stats.skippedUnsupportedTypeTensors());
            assertEquals(0, stats.skippedSmallRowTensors());
            assertEquals(0, stats.skippedCacheTooSmallTensors());
            assertEquals(0, stats.failedTensors());
            assertEquals(392L, stats.preparedBytes());
            assertEquals(3, stats.cacheEntries());
            assertEquals(392L, stats.cacheBytes());
            assertTrue(stats.prepareNanos() >= 0);
            assertTrue(stats.prepareMillis() >= 0.0d);
            assertEquals(3, GgufTensorOps.preparedMatrixCacheSize(model));
            assertEquals(392L, GgufTensorOps.preparedMatrixCacheBytes(model));
            assertEquals(3, GgufTensorOps.clearPreparedMatrixCaches(model));
        }
    }
}
