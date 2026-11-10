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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufQ3KTest {
    @Test
    void supportsQ3KRowDotAndPreparedMatVec() {
        String previous = System.getProperty("aljabr.gguf.q3k.cache_min_rows");
        System.setProperty("aljabr.gguf.q3k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 110);
            writeQ3KBlock(segment.asSlice(0, 110), 1, (byte) 0x55, (byte) 0xFF);
            writeQ3KBlock(segment.asSlice(110, 110), 1, (byte) 0xAA, (byte) 0xFF);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q3_k", new long[]{256, 2}, 11, 0, 2L * 110);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(11));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(512.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q3KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.q3KMatrixCacheBytes(model));

            GgufTensorOps.Q3KMatrix first = GgufTensorOps.q3KMatrixCached(model, tensor);
            GgufTensorOps.Q3KMatrix second = GgufTensorOps.q3KMatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.quantStride());
            assertEquals(16, first.groupStride());
            assertEquals(expectedRowKernel(), first.rowKernel());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ3KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q3k.cache_min_rows", previous);
        }
    }

    private static int expectedRowKernel() {
        return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q3KMatrix.ROW_KERNEL_VECTOR
                : GgufTensorOps.Q3KMatrix.ROW_KERNEL_SCALAR;
    }
}
