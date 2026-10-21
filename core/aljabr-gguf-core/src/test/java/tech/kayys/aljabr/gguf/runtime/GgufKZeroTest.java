package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufKZeroTest {
    @Test
    void preparedAndGenericZeroRowMatVecSkipWorkAfterValidation() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(458);
            writeQ2KMinLaneOrderBlock(segment.asSlice(0, 84));
            writeQ4KMinLaneOrderBlock(segment.asSlice(84, 144));
            writeQ5KMinLaneOrderBlock(segment.asSlice(228, 176));
            writeQ8Block(segment.asSlice(404, 34), (short) 0x3c00, (byte) 2);
            writeQ4_1Block(segment.asSlice(438, 20), (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            GGUFTensorInfo q2Tensor = new GGUFTensorInfo("q2.prepared.zero.mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4.prepared.zero.mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5.prepared.zero.mins", new long[]{256, 1}, 13, 228, 176);
            GGUFTensorInfo q8Tensor = new GGUFTensorInfo("q8.prepared.zero", new long[]{32, 1}, 8, 404, 34);
            GGUFTensorInfo q32Tensor = new GGUFTensorInfo("q32.prepared.zero.bias", new long[]{32, 1}, 3, 438, 20);
            GGUFModel model = new GGUFModel(
                    3,
                    Map.of(),
                    List.of(q2Tensor, q4Tensor, q5Tensor, q8Tensor, q32Tensor),
                    0,
                    segment,
                    null);

            float[] vector = ramp(256);
            float[] output = {-123.0f};
            GgufTensorOps.matVecRows(GgufTensorOps.q2KMatrix(model, q2Tensor), vector, output, 0, true);
            GgufTensorOps.matVecRows(GgufTensorOps.q4KMatrix(model, q4Tensor), vector, output, 0, true);
            GgufTensorOps.matVecRows(GgufTensorOps.q5KMatrix(model, q5Tensor), vector, output, 0, true);
            GgufTensorOps.matVecRows(GgufTensorOps.q8Matrix(model, q8Tensor), vector, output, 0, true);
            GgufTensorOps.matVecRows(GgufTensorOps.q32Matrix(model, q32Tensor), vector, output, 0, true);
            int estimatesBeforeGenericZero = GgufTensorOps.preparedMatrixEstimateCacheSize(model);
            GgufTensorOps.matVecRows(model, q4Tensor, vector, output, 0, true);

            assertEquals(-123.0f, output[0], 0.0f);
            assertEquals(estimatesBeforeGenericZero, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        }
    }
}
