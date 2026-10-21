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

class GgufQ2LaneTest {
    @Test
    void rawQ2KNoMinMatVecPreservesPackedQuantLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q2k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q2k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84);
            writeQ2KNoMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2.raw.no_min.lanes", new long[]{256, 1}, 10, 0, 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            assertEquals(expectedQ2LaneOrderDot(vector, false), output[0], 0.0f);
            assertArrayEquals(expectedQ2LaneOrderRow(false), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q2KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q2k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ2KMinMatVecPreservesPackedQuantLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q2k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q2k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84);
            writeQ2KMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2.raw.min.lanes", new long[]{256, 1}, 10, 0, 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            assertEquals(expectedQ2LaneOrderDot(vector, true), output[0], 0.0f);
            assertArrayEquals(expectedQ2LaneOrderRow(true), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q2KMatrixCacheSize(model));
            assertEquals(384L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q2k.cache_max_bytes", previousMaxBytes);
        }
    }
}
