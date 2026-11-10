package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.assertSingleRowMatVecMatchesDot;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ32OneTest {
    @Test
    void rawQ32MatVecUsesDirectDotPathForSingleDecodeRow() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(18 + 20 + 22 + 24);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_1Block(segment.asSlice(18, 20), (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            writeQ5_0Block(segment.asSlice(38, 22), (short) 0x3c00, 0xFFFF0000, (byte) 0x21);
            writeQ5_1Block(segment.asSlice(60, 24), (short) 0x3c00, (short) 0x3800, -1, (byte) 0x21);
            GGUFTensorInfo q4_0Tensor = new GGUFTensorInfo("q4_0.raw.single.dot", new long[]{32, 1}, 2, 0, 18);
            GGUFTensorInfo q4_1Tensor = new GGUFTensorInfo("q4_1.raw.single.dot", new long[]{32, 1}, 3, 18, 20);
            GGUFTensorInfo q5_0Tensor = new GGUFTensorInfo("q5_0.raw.single.dot", new long[]{32, 1}, 6, 38, 22);
            GGUFTensorInfo q5_1Tensor = new GGUFTensorInfo("q5_1.raw.single.dot", new long[]{32, 1}, 7, 60, 24);
            GGUFModel model = new GGUFModel(
                    3,
                    Map.of(),
                    List.of(q4_0Tensor, q4_1Tensor, q5_0Tensor, q5_1Tensor),
                    0,
                    segment,
                    null);

            float[] vector = ramp(32);
            assertSingleRowMatVecMatchesDot(model, q4_0Tensor, vector);
            assertSingleRowMatVecMatchesDot(model, q4_1Tensor, vector);
            assertSingleRowMatVecMatchesDot(model, q5_0Tensor, vector);
            assertSingleRowMatVecMatchesDot(model, q5_1Tensor, vector);
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previousMaxBytes);
        }
    }
}
