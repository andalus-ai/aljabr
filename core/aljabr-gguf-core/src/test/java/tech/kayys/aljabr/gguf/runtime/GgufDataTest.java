package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufDataTest {
    @Test
    void findTensorUsesPerModelNameIndexAndPreservesFirstMatch() {
        GGUFTensorInfo first = new GGUFTensorInfo("dup.weight", new long[]{2}, 0, 0, 8);
        GGUFTensorInfo second = new GGUFTensorInfo("dup.weight", new long[]{4}, 0, 8, 16);
        GGUFTensorInfo other = new GGUFTensorInfo("other.weight", new long[]{1}, 0, 24, 4);
        GGUFModel model = new GGUFModel(3, Map.of(), List.of(first, second, other), 0, MemorySegment.NULL, null);

        assertSame(first, GgufTensorData.findTensor(model, "dup.weight"));
        assertSame(other, GgufTensorData.findTensor(model, "other.weight"));
        for (int i = 0; i < 4; i++) {
            assertSame(first, GgufTensorData.findTensor(model, "dup.weight"));
            assertSame(other, GgufTensorData.findTensor(model, "other.weight"));
        }
        assertThrows(IllegalArgumentException.class, () -> GgufTensorData.findTensor(model, "missing.weight"));
    }

    @Test
    void tensorDataReusesPerModelTensorSlice() {
        GgufTensorData.clearRecentSliceCache();
        assertEquals(0, GgufTensorData.recentSliceFastCacheSize());
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(64);
            GGUFTensorInfo tensor = new GGUFTensorInfo("weight", new long[]{2, 2}, 0, 8, 16);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 4, segment, null);

            MemorySegment first = GgufTensorData.tensorData(model, tensor);
            assertEquals(1, GgufTensorData.recentSliceCacheSize());
            assertEquals(1, GgufTensorData.recentSliceFastCacheSize());
            MemorySegment second = GgufTensorData.tensorData(model, tensor);

            assertSame(first, second);
            assertEquals(1, GgufTensorData.recentSliceCacheSize());
            assertEquals(1, GgufTensorData.recentSliceFastCacheSize());
            assertEquals(16, first.byteSize());
        } finally {
            GgufTensorData.clearRecentSliceCache();
        }
    }

    @Test
    void tensorDataReusesCachedSlicesWhenAlternatingTensors() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(64);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("first", new long[]{2}, 0, 0, 16);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("second", new long[]{2}, 0, 16, 16);
            GGUFModel model = new GGUFModel(
                    3,
                    Map.of(),
                    List.of(firstTensor, secondTensor),
                    0,
                    segment,
                    null);

            MemorySegment first = GgufTensorData.tensorData(model, firstTensor);
            MemorySegment second = GgufTensorData.tensorData(model, secondTensor);

            for (int i = 0; i < 4; i++) {
                assertSame(first, GgufTensorData.tensorData(model, firstTensor));
                assertSame(second, GgufTensorData.tensorData(model, secondTensor));
            }
        }
    }

    @Test
    void tensorDataKeepsSliceCachesPerModel() {
        GgufTensorData.clearRecentSliceCache();
        try (Arena arena = Arena.ofConfined()) {
            GGUFTensorInfo tensor = new GGUFTensorInfo("weight", new long[]{2, 2}, 0, 0, 16);
            GGUFModel firstModel = new GGUFModel(3, Map.of(), List.of(tensor), 0, arena.allocate(32), null);
            GGUFModel secondModel = new GGUFModel(3, Map.of(), List.of(tensor), 0, arena.allocate(32), null);

            MemorySegment first = GgufTensorData.tensorData(firstModel, tensor);
            MemorySegment second = GgufTensorData.tensorData(secondModel, tensor);
            MemorySegment firstAgain = GgufTensorData.tensorData(firstModel, tensor);

            assertNotSame(first, second);
            assertSame(first, firstAgain);
        } finally {
            GgufTensorData.clearRecentSliceCache();
        }
    }

    @Test
    void tensorDataStillRejectsOutOfRangeSlices() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(16);
            GGUFTensorInfo tensor = new GGUFTensorInfo("bad", new long[]{2, 2}, 0, 8, 16);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 4, segment, null);

            assertThrows(IllegalArgumentException.class, () -> GgufTensorData.tensorData(model, tensor));
        }
    }
}
