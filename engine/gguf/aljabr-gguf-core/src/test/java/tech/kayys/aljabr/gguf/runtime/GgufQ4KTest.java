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

class GgufQ4KTest {
    @Test
    void dequantizesQ4KBlockWithCurrentGgmlScaleLayout() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment block = arena.allocate(144);
            writeSimpleQ4KBlock(block);

            float[] out = new float[256];
            GgufKBlockDequantizer.dequantizeQ4KBlock(block, 0, out, 0);

            for (int superBlock = 0; superBlock < 4; superBlock++) {
                int base = superBlock * 64;
                for (int i = 0; i < 32; i++) {
                    assertEquals(1.0f, out[base + i], 0.0f);
                    assertEquals(2.0f, out[base + 32 + i], 0.0f);
                }
            }
            assertEquals(384.0f, sum(out), 0.0f);
        }
    }

    @Test
    void computesQ4KRowDotWithoutMaterializingTheWholeMatrix() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(144);
            writeSimpleQ4KBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(256);

            assertEquals(384.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
        }
    }
}
