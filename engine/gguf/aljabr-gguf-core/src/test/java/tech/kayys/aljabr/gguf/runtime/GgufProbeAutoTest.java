package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;

import java.lang.foreign.Arena;

import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.decoderCacheModel;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufProbeAutoTest {
    @Test
    void autoPreparesDecoderCachesWhenPlanFitsBudget() {
        try (Arena arena = Arena.ofShared()) {
            GGUFModel model = decoderCacheModel(arena);
            GgufRuntimeProbe.PreparedMatrixCacheSelection selection =
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(model, 0, true, 1, 1024);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 52, 0, 1, 1, selection);

            assertEquals("auto-prepared", selection.mode());
            assertTrue(selection.prepare());
            assertEquals("auto-prepared", probe.preparedMatrixCacheDecision().mode());
            assertTrue(probe.hasPreparedMatrixCacheProbe());
            assertEquals(2, probe.preparedMatrixCacheStats().preparedTensors());
            assertEquals(72L, probe.preparedMatrixCacheStats().cacheBytes());
            assertTrue(probe.preparedMatrixCacheDecision().selectionSummary().contains("budget=0.00MiB"));
            assertEquals(2, GgufTensorOps.clearPreparedMatrixCaches(model));
        }
    }
}
