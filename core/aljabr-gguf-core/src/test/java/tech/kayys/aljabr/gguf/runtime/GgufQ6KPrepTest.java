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

class GgufQ6KPrepTest {
    @Test
    void buildsPreparedQ6KMatrixFromNonZeroTensorOffset() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 210);
            writeQ6KBlock(segment.asSlice(0, 210), (byte) 0x11, (byte) 0xAA, (byte) 1);
            writeQ6KBlock(segment.asSlice(210, 210), (byte) 0x22, (byte) 0xAA, (byte) 1);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6.offset", new long[]{256, 1}, 14, 210, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q6KMatrix matrix = GgufTensorOps.q6KMatrix(model, tensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(matrix, ones(256), output, 1, true);

            assertEquals(512.0f, output[0], 0.0f);
            assertEquals(320L, matrix.estimatedBytes());
        }
    }

    @Test
    void preparedQ6KMatrixPreservesPackedLowAndHighLaneOrder() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(210);
            writeQ6KLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6.prepared.lanes", new long[]{256, 1}, 14, 0, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q6KMatrix matrix = GgufTensorOps.q6KMatrix(model, tensor);
            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(matrix, vector, output, 1, true);

            assertEquals(GgufQ6LaneFx.expectedDot(vector), output[0], 0.0f);
            assertEquals(320L, matrix.estimatedBytes());
        }
    }
}
