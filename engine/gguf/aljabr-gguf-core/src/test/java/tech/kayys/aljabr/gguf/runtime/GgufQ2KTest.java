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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufQ2KTest {
    @Test
    void supportsQ2KRowDotAndPreparedMatVec() {
        String previous = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 84);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeQ2KBlock(segment.asSlice(84, 84), (byte) 0x01, (byte) 0xAA);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2_k", new long[]{256, 2}, 10, 0, 2L * 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(10));

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
            assertEquals(1, GgufTensorOps.q2KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.q2KMatrixCacheBytes(model));

            GgufTensorOps.Q2KMatrix first = GgufTensorOps.q2KMatrixCached(model, tensor);
            GgufTensorOps.Q2KMatrix second = GgufTensorOps.q2KMatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.quantStride());
            assertEquals(16, first.groupStride());
            assertEquals(expectedNoMinKernel(), first.noMinKernel());
            assertEquals(expectedPrecomputedMinsKernel(), first.precomputedMinsKernel());
            assertEquals(expectedDirectMinsKernel(), first.directMinsKernel());
            assertFalse(first.hasGroupMins());
            assertEquals(0, first.groupMins().length);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ2KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previous);
        }
    }

    private static int expectedNoMinKernel() {
        return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q2KMatrix.ROW_KERNEL_NO_MIN_VECTOR
                : GgufTensorOps.Q2KMatrix.ROW_KERNEL_NO_MIN_SCALAR;
    }

    private static int expectedPrecomputedMinsKernel() {
        return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q2KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_VECTOR
                : GgufTensorOps.Q2KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_SCALAR;
    }

    private static int expectedDirectMinsKernel() {
        return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q2KMatrix.ROW_KERNEL_DIRECT_MINS_VECTOR
                : GgufTensorOps.Q2KMatrix.ROW_KERNEL_DIRECT_MINS_SCALAR;
    }
}
