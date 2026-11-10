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
import static org.junit.jupiter.api.Assertions.assertSame;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlock;

class GgufQ5KPrepTest {
    @Test
    void preparesAndCachesQ5KMatrixForGenericMatVec() {
        String previous = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlock(segment.asSlice(0, 176), (byte) 0xFF, (byte) 0);
            writeQ5KBlock(segment.asSlice(176, 176), (byte) 0xFF, (byte) 0x11);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_k", new long[]{256, 2}, 13, 0, 2L * 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(256);
            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(4096.0f, output[0], 0.0f);
            assertEquals(4352.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q5KMatrixCacheSize(model));
            assertEquals(576L, GgufTensorOps.q5KMatrixCacheBytes(model));

            GgufTensorOps.Q5KMatrix first = GgufTensorOps.q5KMatrixCached(model, tensor);
            GgufTensorOps.Q5KMatrix second = GgufTensorOps.q5KMatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.quantStride());
            assertEquals(8, first.groupStride());
            assertEquals(expectedNoMinKernel(), first.noMinKernel());
            assertEquals(expectedPrecomputedMinsKernel(), first.precomputedMinsKernel());
            assertEquals(expectedDirectMinsKernel(), first.directMinsKernel());
            assertFalse(first.hasGroupMins());
            assertEquals(0, first.groupMins().length);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(4096.0f, preparedOutput[0], 0.0f);
            assertEquals(4352.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ5KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previous);
        }
    }

    private static int expectedNoMinKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q5KMatrix.ROW_KERNEL_NO_MIN_VECTOR
                : GgufTensorOps.Q5KMatrix.ROW_KERNEL_NO_MIN_SCALAR;
    }

    private static int expectedPrecomputedMinsKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q5KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_VECTOR
                : GgufTensorOps.Q5KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_SCALAR;
    }

    private static int expectedDirectMinsKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q5KMatrix.ROW_KERNEL_DIRECT_MINS_VECTOR
                : GgufTensorOps.Q5KMatrix.ROW_KERNEL_DIRECT_MINS_SCALAR;
    }
}
