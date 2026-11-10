package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ6KBlock;

class GgufQ6KCacheTest {
    @Test
    void preparesAndCachesQ6KMatrixForGenericMatVec() {
        String previous = System.getProperty("aljabr.gguf.q6k.cache_min_rows");
        System.setProperty("aljabr.gguf.q6k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 210);
            writeQ6KBlock(segment.asSlice(0, 210), (byte) 0x11, (byte) 0xAA, (byte) 1);
            writeQ6KBlock(segment.asSlice(210, 210), (byte) 0x22, (byte) 0xAA, (byte) 1);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6_k", new long[]{256, 2}, 14, 0, 2L * 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(256);
            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q6KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.q6KMatrixCacheBytes(model));

            GgufTensorOps.Q6KMatrix first = GgufTensorOps.q6KMatrixCached(model, tensor);
            GgufTensorOps.Q6KMatrix second = GgufTensorOps.q6KMatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.quantStride());
            assertEquals(16, first.groupStride());
            assertEquals(expectedRowKernel(), first.rowKernel());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ6KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q6k.cache_min_rows", previous);
        }
    }

    private static int expectedRowKernel() {
        return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q6KMatrix.ROW_KERNEL_VECTOR
                : GgufTensorOps.Q6KMatrix.ROW_KERNEL_SCALAR;
    }
}
