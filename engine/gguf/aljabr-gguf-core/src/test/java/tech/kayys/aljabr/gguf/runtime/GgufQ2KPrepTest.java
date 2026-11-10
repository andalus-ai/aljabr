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
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufQ2KPrepTest {
    @Test
    void buildsPreparedQ2KMatrixFromNonZeroTensorOffset() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 84);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeQ2KBlock(segment.asSlice(84, 84), (byte) 0x01, (byte) 0xAA);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2.offset", new long[]{256, 1}, 10, 84, 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q2KMatrix matrix = GgufTensorOps.q2KMatrix(model, tensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(matrix, ones(256), output, 1, true);

            assertEquals(512.0f, output[0], 0.0f);
            assertEquals(320L, matrix.estimatedBytes());
        }
    }

    @Test
    void preparedQ2KMatrixPreservesPackedQuantLaneOrderWithMins() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84);
            writeQ2KMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2.prepared.min.lanes", new long[]{256, 1}, 10, 0, 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q2KMatrix matrix = GgufTensorOps.q2KMatrix(model, tensor);
            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(matrix, vector, output, 1, true);

            float expected = expectedQ2LaneOrderDot(vector, true);
            assertTrue(matrix.hasGroupMins());
            assertEquals(384L, matrix.estimatedBytes());
            assertEquals(expected, output[0], 0.0f);
        }
    }
}
