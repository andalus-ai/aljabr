package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufPreparedMatrixEstimatorTest {
    @Test
    void rememberedEstimateIsReturnedFromModelCache() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.estimate", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufPreparedMatrixEstimator.remember(model, tensor, 288L);

            assertEquals(1, GgufPreparedMatrixEstimator.size(model));
            assertEquals(288L, GgufPreparedMatrixEstimator.estimate(model, tensor));
            assertEquals(1, GgufPreparedMatrixEstimator.size(model));
        }
    }

    @Test
    void keyBasedKMinHintUsesRememberedEstimate() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4.no-mins", new long[]{256, 1}, GgmlType.Q4_K.id, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufKey key = GgufKey.from(tensor);

            GgufPreparedMatrixEstimator.remember(model, tensor, 288L);

            assertEquals(Boolean.FALSE, GgufPreparedMatrixEstimator.cachedKHasMins(model, key, 1, 8));
        }
    }

    @Test
    void keyBasedKMinHintsStayAvailableWhenAlternatingTensors() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(288);
            GGUFTensorInfo withoutMins = new GGUFTensorInfo(
                    "q4.no-mins", new long[]{256, 1}, GgmlType.Q4_K.id, 0, 144);
            GGUFTensorInfo withMins = new GGUFTensorInfo(
                    "q4.with-mins", new long[]{256, 1}, GgmlType.Q4_K.id, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(withoutMins, withMins), 0, segment, null);
            GgufKey withoutMinsKey = GgufKey.from(withoutMins);
            GgufKey withMinsKey = GgufKey.from(withMins);

            GgufPreparedMatrixEstimator.remember(model, withoutMins, 288L);
            GgufPreparedMatrixEstimator.remember(model, withMins, 320L);

            for (int i = 0; i < 4; i++) {
                assertEquals(Boolean.FALSE, GgufPreparedMatrixEstimator.cachedKHasMins(model, withoutMinsKey, 1, 8));
                assertEquals(Boolean.TRUE, GgufPreparedMatrixEstimator.cachedKHasMins(model, withMinsKey, 1, 8));
            }
            assertEquals(2, GgufPreparedMatrixEstimator.size(model));
        }
    }

    @Test
    void keyBasedKMinHintsStayAvailableWhenRotatingManyTensors() {
        GgufPreparedMatrixEstimator.clearRecentEstimateCache();
        assertEquals(0, GgufPreparedMatrixEstimator.recentEstimateFastCacheSize());
        int tensorCount = 6;
        List<GGUFTensorInfo> tensors = rotatingQ4Tensors(tensorCount);
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(tensorCount * 144L);
            GGUFModel model = new GGUFModel(3, Map.of(), tensors, 0, segment, null);
            for (int index = 0; index < tensorCount; index++) {
                long estimate = index % 2 == 0 ? 288L : 320L;
                GgufPreparedMatrixEstimator.remember(model, tensors.get(index), estimate);
            }
            assertEquals(tensorCount, GgufPreparedMatrixEstimator.recentEstimateCacheSize());
            assertEquals(1, GgufPreparedMatrixEstimator.recentEstimateFastCacheSize());

            for (int pass = 0; pass < 4; pass++) {
                for (int index = 0; index < tensorCount; index++) {
                    GgufKey key = GgufKey.from(tensors.get(index));
                    Boolean expected = index % 2 == 0 ? Boolean.FALSE : Boolean.TRUE;
                    assertEquals(expected, GgufPreparedMatrixEstimator.cachedKHasMins(model, key, 1, 8));
                }
            }
            assertEquals(tensorCount, GgufPreparedMatrixEstimator.recentEstimateCacheSize());
            assertEquals(1, GgufPreparedMatrixEstimator.recentEstimateFastCacheSize());
            assertEquals(tensorCount, GgufPreparedMatrixEstimator.size(model));
        } finally {
            GgufPreparedMatrixEstimator.clearRecentEstimateCache();
        }
    }

    private static List<GGUFTensorInfo> rotatingQ4Tensors(int count) {
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        boolean[] slots = new boolean[256];
        int candidate = 0;
        while (tensors.size() < count) {
            int index = tensors.size();
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4.estimate.rotate." + candidate,
                    new long[]{256, 1},
                    GgmlType.Q4_K.id,
                    index * 144L,
                    144);
            int slot = GgufKey.from(tensor).hashCode() & 255;
            if (!slots[slot]) {
                slots[slot] = true;
                tensors.add(tensor);
            }
            candidate++;
        }
        return tensors;
    }
}
