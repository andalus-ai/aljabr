package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KBlockWithAllScalesAndMins;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeSimpleQ4KBlock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

class GgufQ4KPrepTest {
    @Test
    void genericQ4KMatVecCanUsePreparedCachePath() {
        String previous = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];

            GgufTensorOps.matVecRows(model, tensor, ones(256), output, 2, true);

            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(128.0f, output[1], 0.0f);
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previous);
        }
    }

    @Test
    void reusesPreparedQ4KMatrixPerModelAndTensor() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix first = GgufTensorOps.q4KMatrixCached(model, tensor);
            GgufTensorOps.Q4KMatrix second = GgufTensorOps.q4KMatrixCached(model, tensor);

            assertSame(first, second);
            assertEquals(256, first.quantStride());
            assertEquals(8, first.groupStride());
            assertEquals(expectedNoMinKernel(), first.noMinKernel());
            assertEquals(expectedPrecomputedMinsKernel(), first.precomputedMinsKernel());
            assertEquals(expectedDirectMinsKernel(), first.directMinsKernel());
            assertEquals(1, GgufTensorOps.clearQ4KMatrixCache(model));
            GgufTensorOps.Q4KMatrix afterClear = GgufTensorOps.q4KMatrixCached(model, tensor);
            assertNotSame(first, afterClear);
            assertEquals(1, GgufTensorOps.clearQ4KMatrixCache(model));
        }
    }

    @Test
    void buildsPreparedQ4KMatrixFromNonZeroTensorOffset() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(288);
            writeSimpleQ4KBlock(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.offset", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrix(model, tensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(matrix, ones(256), output, 1, true);

            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(320L, matrix.estimatedBytes());
        }
    }

    private static int expectedNoMinKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q4KMatrix.ROW_KERNEL_NO_MIN_VECTOR
                : GgufTensorOps.Q4KMatrix.ROW_KERNEL_NO_MIN_SCALAR;
    }

    private static int expectedPrecomputedMinsKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q4KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_VECTOR
                : GgufTensorOps.Q4KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_SCALAR;
    }

    private static int expectedDirectMinsKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q4KMatrix.ROW_KERNEL_DIRECT_MINS_VECTOR
                : GgufTensorOps.Q4KMatrix.ROW_KERNEL_DIRECT_MINS_SCALAR;
    }
}
