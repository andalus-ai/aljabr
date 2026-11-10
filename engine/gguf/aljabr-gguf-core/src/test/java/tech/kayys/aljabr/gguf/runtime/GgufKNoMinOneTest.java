package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.assertSingleRowMatVecMatchesDot;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KNoMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KNoMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KNoMinLaneOrderBlock;

class GgufKNoMinOneTest {
    @Test
    void rawNoMinKMatVecUsesDirectDotPathForSingleDecodeRow() {
        String previousQ2MinRows = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        String previousQ2MaxBytes = System.getProperty("aljabr.gguf.q2k.cache_max_bytes");
        String previousQ4MinRows = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        String previousQ4MaxBytes = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        String previousQ5MinRows = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        String previousQ5MaxBytes = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q2k.cache_max_bytes", "1");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "1");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84 + 144 + 176);
            writeQ2KNoMinLaneOrderBlock(segment.asSlice(0, 84));
            writeQ4KNoMinLaneOrderBlock(segment.asSlice(84, 144));
            writeQ5KNoMinLaneOrderBlock(segment.asSlice(228, 176));
            GGUFTensorInfo q2Tensor = new GGUFTensorInfo("q2.raw.single.no_mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4.raw.single.no_mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5.raw.single.no_mins", new long[]{256, 1}, 13, 228, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q2Tensor, q4Tensor, q5Tensor), 0, segment, null);

            float[] vector = ramp(256);
            assertSingleRowMatVecMatchesDot(model, q2Tensor, vector);
            assertSingleRowMatVecMatchesDot(model, q4Tensor, vector);
            assertSingleRowMatVecMatchesDot(model, q5Tensor, vector);
            assertEquals(0, GgufTensorOps.q2KMatrixCacheSize(model));
            assertEquals(0, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(0, GgufTensorOps.q5KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previousQ2MinRows);
            restoreProperty("aljabr.gguf.q2k.cache_max_bytes", previousQ2MaxBytes);
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previousQ4MinRows);
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previousQ4MaxBytes);
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previousQ5MinRows);
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previousQ5MaxBytes);
        }
    }
}
