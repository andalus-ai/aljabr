package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KBlockWithAllScalesAndMins;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeSimpleQ4KBlock;

class GgufQ4KWorkTest {
    @Test
    void supportsReusableQ4KWorkBufferForPreparedMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrix(model, tensor);
            GgufTensorOps.Q4KWorkBuffer workBuffer = new GgufTensorOps.Q4KWorkBuffer();

            float[] output = new float[2];

            GgufTensorOps.matVecRows(matrix, ones(256), output, 2, true, workBuffer);
            int firstCapacity = workBuffer.vectorGroupSumCapacity();
            GgufTensorOps.matVecRows(matrix, ones(256), output, 2, true, workBuffer);

            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(128.0f, output[1], 0.0f);
            assertTrue(firstCapacity >= 8);
            assertEquals(firstCapacity, workBuffer.vectorGroupSumCapacity());
            workBuffer.clear();
            assertEquals(0, workBuffer.vectorGroupSumCapacity());
        }
    }

    @Test
    void reusableWorkBufferGrowsWithHeadroom() {
        GgufTensorOps.Q4KWorkBuffer workBuffer = new GgufTensorOps.Q4KWorkBuffer();
        float[] vector = ones(512);

        GgufSum.vector32GroupSums(vector, 288, workBuffer);
        assertEquals(16, workBuffer.vectorGroupSumCapacity());

        GgufSum.vector32GroupSums(vector, 320, workBuffer);
        assertEquals(16, workBuffer.vectorGroupSumCapacity());

        GgufSum.vector32GroupSums(vector, 512, workBuffer);
        assertEquals(16, workBuffer.vectorGroupSumCapacity());
    }

    @Test
    void defaultPreparedQ4KPathLazilyUsesWorkBufferOnlyWhenMinsNeedGroupSums() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeSimpleQ4KBlock(segment.asSlice(0, 144));
            writeSimpleQ4KBlock(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.no_mins", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrix(model, tensor);
            float[] output = new float[2];

            GgufPrepRows.q4K(matrix, ones(256), output, 2, true, null);

            assertEquals(384.0f, output[0], 0.0f);
            assertEquals(384.0f, output[1], 0.0f);
        }
    }

    @Test
    void publicReusableWorkBufferOverloadStillRequiresBuffer() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeSimpleQ4KBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.no_mins", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrix(model, tensor);

            assertThrows(
                    NullPointerException.class,
                    () -> GgufTensorOps.matVecRows(matrix, ones(256), new float[1], 1, true, null));
        }
    }
}
