package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;

class GgufQ8SignedTest {
    @Test
    void preparedQ8MatrixPreservesSignedQuantValues() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(34);
            writeQ8Block(segment, (short) 0x3c00, (byte) -2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8.signed", new long[]{32, 1}, 8, 0, 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(32);
            assertEquals(-64.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(GgufTensorOps.q8Matrix(model, tensor), vector, output, 1, true);
            assertEquals(-64.0f, output[0], 0.0f);
        }
    }
}
