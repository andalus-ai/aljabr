package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufKOneTest {
    @Test
    void rawQ3KAndQ6KMatVecUseDirectDotPathForSingleDecodeRow() {
        String previousQ3MinRows = System.getProperty("aljabr.gguf.q3k.cache_min_rows");
        String previousQ3MaxBytes = System.getProperty("aljabr.gguf.q3k.cache_max_bytes");
        String previousQ6MinRows = System.getProperty("aljabr.gguf.q6k.cache_min_rows");
        String previousQ6MaxBytes = System.getProperty("aljabr.gguf.q6k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q3k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q3k.cache_max_bytes", "1");
        System.setProperty("aljabr.gguf.q6k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q6k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(110 + 210);
            writeQ3KLaneOrderBlock(segment.asSlice(0, 110));
            writeQ6KLaneOrderBlock(segment.asSlice(110, 210));
            GGUFTensorInfo q3Tensor = new GGUFTensorInfo("q3.raw.single.dot", new long[]{256, 1}, 11, 0, 110);
            GGUFTensorInfo q6Tensor = new GGUFTensorInfo("q6.raw.single.dot", new long[]{256, 1}, 14, 110, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q3Tensor, q6Tensor), 0, segment, null);

            float[] vector = ramp(256);
            assertSingleRowMatVecMatchesDot(model, q3Tensor, vector);
            assertSingleRowMatVecMatchesDot(model, q6Tensor, vector);
            assertEquals(0, GgufTensorOps.q3KMatrixCacheSize(model));
            assertEquals(0, GgufTensorOps.q6KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q3k.cache_min_rows", previousQ3MinRows);
            restoreProperty("aljabr.gguf.q3k.cache_max_bytes", previousQ3MaxBytes);
            restoreProperty("aljabr.gguf.q6k.cache_min_rows", previousQ6MinRows);
            restoreProperty("aljabr.gguf.q6k.cache_max_bytes", previousQ6MaxBytes);
        }
    }

    @Test
    void rawQ4KAndQ5KMatVecRememberHintsWhenEstimateHintIsMissing() {
        String previousQ4MinRows = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        String previousQ5MinRows = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "32");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "32");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144 + 176);
            writeQ4KMinLaneOrderBlock(segment.asSlice(0, 144));
            writeQ5KMinLaneOrderBlock(segment.asSlice(144, 176));
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4.raw.unknown_mins", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5.raw.unknown_mins", new long[]{256, 1}, 13, 144, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q4Tensor, q5Tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] q4Output = new float[1];
            float[] q5Output = new float[1];
            GgufTensorOps.matVecRows(model, q4Tensor, vector, q4Output, 1, true);
            GgufTensorOps.matVecRows(model, q5Tensor, vector, q5Output, 1, true);

            assertEquals(GgufTensorOps.dotRow(model, q4Tensor, 0, vector), q4Output[0], 0.0f);
            assertEquals(GgufTensorOps.dotRow(model, q5Tensor, 0, vector), q5Output[0], 0.0f);
            assertEquals(0, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(0, GgufTensorOps.q5KMatrixCacheSize(model));
            assertEquals(2, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previousQ4MinRows);
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previousQ5MinRows);
        }
    }
}
