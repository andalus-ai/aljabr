package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufPrepPlanTest {
    @Test
    void plansPreparedMatrixCachesWithoutAllocating() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(1);
            GGUFModel model = GgufPrepFx.mixedPreparedPlanModel(segment);

            GgufTensorOps.PreparedMatrixCachePlan plan = GgufTensorOps.planPreparedMatrixCaches(model);

            assertTrue(plan.ready());
            assertEquals(5, plan.scannedTensors());
            assertEquals(4, plan.matrixTensors());
            assertEquals(3, plan.preparedCandidates());
            assertEquals(1, plan.skippedUnsupportedTypeTensors());
            assertEquals(0, plan.skippedSmallRowTensors());
            assertEquals(0, plan.skippedCacheTooSmallTensors());
            assertEquals(0, plan.failedTensors());
            assertEquals(392L, plan.estimatedPreparedBytes());
            assertEquals(0, GgufTensorOps.preparedMatrixCacheSize(model));
            assertEquals(0L, GgufTensorOps.preparedMatrixCacheBytes(model));
            assertTrue(plan.compactSummary().contains("candidates=3/4"));
            assertTrue(plan.compactSummary().contains("skippedCacheTooSmall=0"));
        }
    }
}
