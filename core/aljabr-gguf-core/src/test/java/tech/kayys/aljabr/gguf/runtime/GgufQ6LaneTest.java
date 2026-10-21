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
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ6KLaneOrderBlock;

class GgufQ6LaneTest {
    @Test
    void rawQ6KMatVecPreservesPackedLowAndHighLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q6k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q6k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q6k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q6k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(210);
            writeQ6KLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6.raw.lanes", new long[]{256, 1}, 14, 0, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            assertEquals(GgufQ6LaneFx.expectedDot(vector), output[0], 0.0f);
            assertArrayEquals(GgufQ6LaneFx.expectedRow(), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q6KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q6k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q6k.cache_max_bytes", previousMaxBytes);
        }
    }
}
