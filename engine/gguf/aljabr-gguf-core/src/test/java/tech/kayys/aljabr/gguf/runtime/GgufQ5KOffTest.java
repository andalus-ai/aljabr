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
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlock;

class GgufQ5KOffTest {
    @Test
    void buildsPreparedQ5KMatrixFromNonZeroTensorOffset() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlock(segment.asSlice(0, 176), (byte) 0xFF, (byte) 0);
            writeQ5KBlock(segment.asSlice(176, 176), (byte) 0xFF, (byte) 0x11);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5.offset", new long[]{256, 1}, 13, 176, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q5KMatrix matrix = GgufTensorOps.q5KMatrix(model, tensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(matrix, ones(256), output, 1, true);

            assertEquals(4352.0f, output[0], 0.0f);
            assertEquals(288L, matrix.estimatedBytes());
        }
    }
}
