package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;

class GgufQ32ZeroTest {
    @Test
    void zeroBiasQ32MatricesUseCompactPreparedAndRawFallbackEstimates() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(20 + 24);
            writeQ4_1Block(segment.asSlice(0, 20), (short) 0x3c00, (short) 0, (byte) 0x21);
            writeQ5_1Block(segment.asSlice(20, 24), (short) 0x3c00, (short) 0, -1, (byte) 0x21);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4_1.zero_bias", new long[]{32, 1}, 3, 0, 20);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5_1.zero_bias", new long[]{32, 1}, 7, 20, 24);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q4Tensor, q5Tensor), 0, segment, null);

            assertEquals(36L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q4Tensor));
            assertEquals(36L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q5Tensor));

            float[] vector = ones(32);
            float[] q4Output = new float[1];
            float[] q5Output = new float[1];
            GgufTensorOps.matVecRows(model, q4Tensor, vector, q4Output, 1, true);
            GgufTensorOps.matVecRows(model, q5Tensor, vector, q5Output, 1, true);

            assertEquals(48.0f, q4Output[0], 0.0f);
            assertEquals(560.0f, q5Output[0], 0.0f);
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(2, GgufTensorOps.preparedMatrixEstimateCacheSize(model));

            GgufTensorOps.Q32Matrix q4Matrix = GgufTensorOps.q32Matrix(model, q4Tensor);
            GgufTensorOps.Q32Matrix q5Matrix = GgufTensorOps.q32Matrix(model, q5Tensor);
            assertFalse(q4Matrix.hasBlockBiases());
            assertFalse(q5Matrix.hasBlockBiases());
            assertEquals(0, q4Matrix.blockBiases().length);
            assertEquals(0, q5Matrix.blockBiases().length);
            assertEquals(36L, q4Matrix.estimatedBytes());
            assertEquals(36L, q5Matrix.estimatedBytes());
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previousMaxBytes);
        }
    }
}
