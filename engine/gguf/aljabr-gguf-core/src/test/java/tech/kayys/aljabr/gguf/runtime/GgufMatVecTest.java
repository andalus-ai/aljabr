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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static tech.kayys.aljabr.gguf.runtime.GgufDenseTest.LE_FLOAT;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeSimpleQ4KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;

class GgufMatVecTest {
    @Test
    void rotatingMatVecCallsReuseRecentDispatchPlans() {
        GgufMatVec.clearRecentPlanCache();
        assertEquals(0, GgufMatVec.recentPlanFastCacheSize());
        int tensorCount = 6;
        int columns = 4;
        List<GGUFTensorInfo> tensors = rotatingF32Tensors(tensorCount, columns);
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(tensorCount * columns * (long) Float.BYTES);
            for (int index = 0; index < tensorCount * columns; index++) {
                segment.set(LE_FLOAT, index * (long) Float.BYTES, 1.0f);
            }
            GGUFModel model = new GGUFModel(3, Map.of(), tensors, 0, segment, null);
            float[] vector = ones(columns);
            float[] output = new float[1];

            for (GGUFTensorInfo tensor : tensors) {
                GgufTensorOps.matVecRows(model, tensor, vector, output, 1, false);
                assertEquals(columns, output[0], 0.0f);
            }
            assertEquals(tensorCount, GgufMatVec.recentPlanCacheSize());
            assertEquals(1, GgufMatVec.recentPlanFastCacheSize());

            for (int pass = 0; pass < 4; pass++) {
                for (GGUFTensorInfo tensor : tensors) {
                    GgufTensorOps.matVecRows(model, tensor, vector, output, 1, false);
                    assertEquals(columns, output[0], 0.0f);
                }
            }
            assertEquals(tensorCount, GgufMatVec.recentPlanCacheSize());
            assertEquals(1, GgufMatVec.recentPlanFastCacheSize());
            assertThrows(
                    IllegalArgumentException.class,
                    () -> GgufTensorOps.matVecRows(model, tensors.get(0), ones(columns - 1), output, 1, false));
        } finally {
            GgufMatVec.clearRecentPlanCache();
        }
    }

    @Test
    void preparedMatVecPlanCarriesStableKeyAcrossRotation() {
        GgufMatVec.clearRecentPlanCache();
        int tensorCount = 4;
        List<GGUFTensorInfo> tensors = rotatingQ4KTensors(tensorCount);
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(tensorCount * 144L);
            for (int index = 0; index < tensorCount; index++) {
                writeSimpleQ4KBlock(segment.asSlice(index * 144L, 144));
            }
            GGUFModel model = new GGUFModel(3, Map.of(), tensors, 0, segment, null);
            float[] vector = ones(256);
            float[] output = new float[1];

            for (int pass = 0; pass < 4; pass++) {
                for (GGUFTensorInfo tensor : tensors) {
                    GgufTensorOps.matVecRows(model, tensor, vector, output, 1, false);
                    assertEquals(384.0f, output[0], 0.0f);
                }
            }
            assertEquals(tensorCount, GgufMatVec.recentPlanCacheSize());
        } finally {
            GgufMatVec.clearRecentPlanCache();
        }
    }

    @Test
    void smallNoHintPreparedFamilyMatVecDoesNotCreateCacheKey() {
        String previousMinRows = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "32");
        GgufMatVec.clearRecentPlanCache();
        GgufKey.clearCaches();
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(34);
            writeQ8Block(segment, (short) 0x3c00, (byte) 1);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q8.small.no_key",
                    new long[]{32, 1},
                    GgmlType.Q8_0.id,
                    0,
                    34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(model, tensor, ones(32), output, 1, false);

            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(0, GgufKey.recentKeyCacheSize());
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previousMinRows);
            GgufMatVec.clearRecentPlanCache();
            GgufKey.clearCaches();
        }
    }

    @Test
    void admittedNoHintPreparedFamilyMatVecCreatesCacheKeyLazily() {
        String previousMinRows = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "2");
        GgufMatVec.clearRecentPlanCache();
        GgufKey.clearCaches();
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(68);
            writeQ8Block(segment.asSlice(0, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(34, 34), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q8.admitted.lazy_key",
                    new long[]{32, 2},
                    GgmlType.Q8_0.id,
                    0,
                    68);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            float[] output = new float[2];

            GgufTensorOps.matVecRows(model, tensor, ones(32), output, 2, false);

            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(64.0f, output[1], 0.0f);
            assertEquals(1, GgufKey.recentKeyCacheSize());
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previousMinRows);
            GgufMatVec.clearRecentPlanCache();
            GgufKey.clearCaches();
        }
    }

    private static List<GGUFTensorInfo> rotatingF32Tensors(int count, int columns) {
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        boolean[] slots = new boolean[256];
        int candidate = 0;
        while (tensors.size() < count) {
            int index = tensors.size();
            long offset = index * columns * (long) Float.BYTES;
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "f32.matvec.rotate." + candidate,
                    new long[]{columns, 1},
                    GgmlType.F32.id,
                    offset,
                    columns * (long) Float.BYTES);
            int slot = System.identityHashCode(tensor) & 255;
            if (!slots[slot]) {
                slots[slot] = true;
                tensors.add(tensor);
            }
            candidate++;
        }
        return tensors;
    }

    private static List<GGUFTensorInfo> rotatingQ4KTensors(int count) {
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        boolean[] slots = new boolean[256];
        int candidate = 0;
        while (tensors.size() < count) {
            int index = tensors.size();
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4.matvec.rotate." + candidate,
                    new long[]{256, 1},
                    GgmlType.Q4_K.id,
                    index * 144L,
                    144);
            int slot = System.identityHashCode(tensor) & 255;
            if (!slots[slot]) {
                slots[slot] = true;
                tensors.add(tensor);
            }
            candidate++;
        }
        return tensors;
    }
}
