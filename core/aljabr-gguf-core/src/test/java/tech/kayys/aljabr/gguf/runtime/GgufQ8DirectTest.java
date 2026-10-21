package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.*;

class GgufQ8DirectTest {
    @Test
    void rawQ8FamilyMatVecUsesDirectDotPathForSingleDecodeRow() {
        String previousMinRows = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q8.cache_max_bytes");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q8.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(18 + 17 + 34 + 292);
            writeQ1_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0xFF);
            writeMXFP4Block(segment.asSlice(18, 17), (byte) 128, (byte) 0xA5);
            writeQ8Block(segment.asSlice(35, 34), (short) 0x3c00, (byte) 2);
            writeQ8KBlock(segment.asSlice(69, 292), 1.0f, (byte) 2);
            GGUFTensorInfo q1Tensor = new GGUFTensorInfo("q1.raw.single", new long[]{128, 1}, 41, 0, 18);
            GGUFTensorInfo mxfp4Tensor = new GGUFTensorInfo("mxfp4.raw.single", new long[]{32, 1}, 39, 18, 17);
            GGUFTensorInfo q8Tensor = new GGUFTensorInfo("q8.raw.single", new long[]{32, 1}, 8, 35, 34);
            GGUFTensorInfo q8KTensor = new GGUFTensorInfo("q8_k.raw.single", new long[]{256, 1}, 15, 69, 292);
            GGUFModel model = new GGUFModel(
                    3,
                    Map.of(),
                    List.of(q1Tensor, mxfp4Tensor, q8Tensor, q8KTensor),
                    0,
                    segment,
                    null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            assertDirectDotPath(model, q1Tensor, vector, output);
            assertDirectDotPath(model, mxfp4Tensor, vector, output);
            assertDirectDotPath(model, q8Tensor, vector, output);
            assertDirectDotPath(model, q8KTensor, vector, output);

            assertEquals(0, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(4, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q8.cache_max_bytes", previousMaxBytes);
        }
    }

    private static void assertDirectDotPath(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            float[] output) {
        GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
        assertEquals(GgufTensorOps.dotRow(model, tensor, 0, vector), output[0], 0.0f);
    }
}
