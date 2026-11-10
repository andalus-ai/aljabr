package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.*;

class GgufQ5LaneTest {
    @Test
    void rawQ5KNoMinMatVecPreservesPackedNibbleAndHighBitLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(176);
            writeQ5KNoMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5.raw.no_min.lanes", new long[]{256, 1}, 13, 0, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            float expected = expectedQ5LaneOrderDot(vector, false);
            assertEquals(expected, output[0], 0.0f);
            assertEquals(expected, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertArrayEquals(expectedQ5LaneOrderRow(false), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q5KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ5KMinMatVecPreservesPackedNibbleAndHighBitLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(176);
            writeQ5KMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5.raw.min.lanes", new long[]{256, 1}, 13, 0, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            float expected = expectedQ5LaneOrderDot(vector, true);
            assertEquals(expected, output[0], 0.0f);
            assertEquals(expected, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertArrayEquals(expectedQ5LaneOrderRow(true), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q5KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previousMaxBytes);
        }
    }
}
