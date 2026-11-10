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

class GgufQ3KPrepTest {
    @Test
    void buildsPreparedQ3KMatrixFromNonZeroTensorOffset() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 110);
            writeQ3KBlock(segment.asSlice(0, 110), 1, (byte) 0x55, (byte) 0xFF);
            writeQ3KBlock(segment.asSlice(110, 110), 1, (byte) 0xAA, (byte) 0xFF);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q3.offset", new long[]{256, 1}, 11, 110, 110);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q3KMatrix matrix = GgufTensorOps.q3KMatrix(model, tensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(matrix, ones(256), output, 1, true);

            assertEquals(512.0f, output[0], 0.0f);
            assertEquals(320L, matrix.estimatedBytes());
        }
    }

    @Test
    void preparedQ3KMatrixPreservesPackedQuantAndHighMaskLaneOrder() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(110);
            writeQ3KLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q3.prepared.lanes", new long[]{256, 1}, 11, 0, 110);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q3KMatrix matrix = GgufTensorOps.q3KMatrix(model, tensor);
            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(matrix, vector, output, 1, true);

            assertEquals(expectedQ3LaneOrderDot(vector), output[0], 0.0f);
            assertEquals(320L, matrix.estimatedBytes());
        }
    }
}
