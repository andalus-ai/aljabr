package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ6KBlock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufQ6KTest {
    @Test
    void supportsQ6KRowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(210);
            writeQ6KBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6_k", new long[]{256, 1}, 14, 0, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(14));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
            assertEquals(256.0f, output[0], 0.0f);
        }
    }
}
