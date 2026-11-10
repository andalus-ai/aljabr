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
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeSimpleQ4KBlock;

class GgufQ4KVecTest {
    @Test
    void computesQ4KMatVecRowsInParallel() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeSimpleQ4KBlock(segment.asSlice(0, 144));
            writeSimpleQ4KBlock(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(256);
            float[] output = new float[2];

            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(384.0f, output[0], 0.0f);
            assertEquals(384.0f, output[1], 0.0f);

            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrix(model, tensor);
            assertFalse(matrix.hasGroupMins());
            assertEquals(0, matrix.groupMins().length);
            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(matrix, vector, preparedOutput, 2, true);
            assertEquals(384.0f, preparedOutput[0], 0.0f);
            assertEquals(384.0f, preparedOutput[1], 0.0f);
        }
    }
}
