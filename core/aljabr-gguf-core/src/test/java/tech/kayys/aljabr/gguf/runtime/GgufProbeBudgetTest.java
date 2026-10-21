package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;

import java.lang.foreign.Arena;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.decoderCacheModel;

class GgufProbeBudgetTest {
    @Test
    void autoPreparesBudgetedDecoderCacheSubsetWhenFullPlanExceedsBudget() {
        try (Arena arena = Arena.ofShared()) {
            GGUFModel model = decoderCacheModel(arena);
            GgufRuntimeProbe.PreparedMatrixCacheSelection selection =
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(model, 0, true, 1, 40);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 52, 0, 1, 1, selection);

            assertEquals("auto-budget-prepared", selection.mode());
            assertTrue(selection.prepare());
            assertEquals(1, selection.plan().preparedCandidates());
            assertEquals(36L, selection.plan().estimatedPreparedBytes());
            assertEquals(1, selection.selectedTensors().size());
            assertEquals("blk.0.attn_q.weight", selection.selectedTensors().get(0).name());
            assertEquals("auto-budget-prepared", probe.preparedMatrixCacheDecision().mode());
            assertTrue(probe.hasPreparedMatrixCacheProbe());
            assertEquals(1, probe.preparedMatrixCacheStats().preparedTensors());
            assertEquals(1, probe.preparedMatrixCacheStats().cacheEntries());
            assertEquals(36L, probe.preparedMatrixCacheStats().cacheBytes());
            assertTrue(probe.preparedMatrixCacheDecision().selectionSummary().contains("budget=0.00MiB"));
            assertEquals(1, GgufTensorOps.clearPreparedMatrixCaches(model));
        }
    }

    @Test
    void autoSkipsDecoderCachesWhenPlanExceedsBudget() {
        try (Arena arena = Arena.ofShared()) {
            GGUFModel model = decoderCacheModel(arena);
            GgufRuntimeProbe.PreparedMatrixCacheSelection selection =
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(model, 0, true, 1, 1);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 52, 0, 1, 1, selection);

            assertEquals("auto-skipped-budget", selection.mode());
            assertFalse(selection.prepare());
            assertEquals("auto-skipped-budget", probe.preparedMatrixCacheDecision().mode());
            assertTrue(probe.hasPreparedMatrixCachePlan());
            assertFalse(probe.hasPreparedMatrixCacheProbe());
            assertEquals(72L, probe.preparedMatrixCachePlan().estimatedPreparedBytes());
            assertEquals(0, probe.preparedMatrixCacheStats().preparedTensors());
            GgufTensorOps.clearPreparedMatrixCaches(model);
        }
    }
}
