package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;

import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.decoderCacheModel;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufRuntimeProbeCacheTest {
    @Test
    void probesDecoderPreparedMatrixCacheCoverageWhenRequested() {
        try (Arena arena = Arena.ofShared()) {
            GGUFModel model = decoderCacheModel(arena);
            GGUFTensorInfo q4 = model.tensors().get(0);
            GGUFTensorInfo ignored = model.tensors().get(2);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 52, 1, 1, 1, 1);

            assertTrue(probe.hasTensorProbe());
            assertTrue(probe.hasPreparedMatrixCachePlan());
            assertTrue(probe.hasPreparedMatrixCacheProbe());
            assertEquals("blk.0.attn_q.weight", probe.tensorName());
            GgufTensorOps.PreparedMatrixCachePlan plan = probe.preparedMatrixCachePlan();
            assertTrue(plan.ready());
            assertEquals(2, plan.scannedTensors());
            assertEquals(2, plan.matrixTensors());
            assertEquals(2, plan.preparedCandidates());
            assertEquals(72L, plan.estimatedPreparedBytes());
            GgufTensorOps.PreparedMatrixCacheStats stats = probe.preparedMatrixCacheStats();
            assertTrue(stats.ready());
            assertEquals(2, stats.scannedTensors());
            assertEquals(2, stats.matrixTensors());
            assertEquals(2, stats.preparedCandidates());
            assertEquals(2, stats.preparedTensors());
            assertEquals(0, stats.failedTensors());
            assertEquals(72L, stats.preparedBytes());
            assertEquals(2, stats.cacheEntries());
            assertEquals(72L, stats.cacheBytes());
            assertEquals("explicit-prepared", probe.preparedMatrixCacheDecision().mode());
            assertTrue(probe.preparedMatrixCacheDecision().compactSummary().contains("mode=explicit-prepared"));
            assertTrue(GgufRuntimeProbe.isDecoderMatVecWeight(q4));
            assertFalse(GgufRuntimeProbe.isDecoderMatVecWeight(ignored));
            assertTrue(stats.compactSummary().contains("prepared=2/2"));
            assertEquals(2, GgufTensorOps.clearPreparedMatrixCaches(model));
        }
    }

    @Test
    void plansDecoderPreparedMatrixCacheCoverageWithoutPreparingWhenDisabled() {
        try (Arena arena = Arena.ofShared()) {
            GGUFModel model = decoderCacheModel(arena);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 52, 1, 1);

            assertTrue(probe.hasTensorProbe());
            assertTrue(probe.hasPreparedMatrixCachePlan());
            assertFalse(probe.hasPreparedMatrixCacheProbe());
            GgufTensorOps.PreparedMatrixCachePlan plan = probe.preparedMatrixCachePlan();
            assertTrue(plan.ready());
            assertEquals(2, plan.scannedTensors());
            assertEquals(2, plan.matrixTensors());
            assertEquals(2, plan.preparedCandidates());
            assertEquals(72L, plan.estimatedPreparedBytes());
            assertEquals(0, probe.preparedMatrixCacheStats().scannedTensors());
            assertTrue(plan.compactSummary().contains("estimatedBytes="));
            assertEquals("auto-skipped-disabled", probe.preparedMatrixCacheDecision().mode());
            GgufTensorOps.clearPreparedMatrixCaches(model);
        }
    }
}
