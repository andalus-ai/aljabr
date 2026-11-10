package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8KBlock;

class GgufQ8KTest {
    private static final int VECTOR_OFFSET = 7;

    @Test
    void supportsQ8KRowDotAndPreparedMatVec() {
        String previous = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 292);
            writeQ8KBlock(segment.asSlice(0, 292), 1.0f, (byte) 1);
            writeQ8KBlock(segment.asSlice(292, 292), 1.0f, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8_k", new long[]{256, 2}, 15, 0, 2L * 292);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(15));

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
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(520L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.blockSize());
            assertEquals(256, first.quantStride());
            assertEquals(GgufTensorOps.Q8Matrix.ROW_ROUTE_WIDE, first.rowRoute());
            assertEquals(
                    GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                            ? GgufTensorOps.Q8Matrix.ROW_KERNEL_WIDE_VECTOR
                            : GgufTensorOps.Q8Matrix.ROW_KERNEL_WIDE_SCALAR,
                    first.rowKernel());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void rawQ8KVectorAccumulatorMatchesScalarReference() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(5L * Q8_K_BLOCK_BYTES);
            float[] scales = {1.25f, -0.75f, 0.5f, 1.75f, -1.125f};
            int[] seeds = {19, 91, 37, 113, 157};
            for (int block = 0; block < scales.length; block++) {
                writePatternQ8KBlock(
                        segment.asSlice(block * (long) Q8_K_BLOCK_BYTES, Q8_K_BLOCK_BYTES),
                        scales[block],
                        seeds[block]);
            }

            int columns = scales.length * QK_K;
            float[] vector = new float[columns + VECTOR_OFFSET];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 17 - 8) * 0.125f;
            }

            float expected = referenceRawQ8K(scales, seeds, vector, VECTOR_OFFSET);
            float scalar = GgufQ8RawDot.dotRowQ8KScalar(segment, 0, columns, vector, VECTOR_OFFSET);
            float vectorized = GgufQ8RawDot.dotRowQ8KVector(segment, 0, columns, vector, VECTOR_OFFSET);

            assertEquals(expected, scalar, 1.0e-3f);
            assertEquals(expected, vectorized, 1.0e-3f);
        }
    }

    @Test
    void preparedQ8KWideVectorAccumulatorMatchesIndependentReferenceAcrossBlocks() {
        int blocks = 5;
        byte[] quants = new byte[blocks * QK_K];
        float[] scales = {1.25f, -0.75f, 0.5f, 1.5f, -1.0f};
        int[] seeds = {23, 67, 113, 151, 193};
        for (int block = 0; block < blocks; block++) {
            for (int index = 0; index < QK_K; index++) {
                quants[block * QK_K + index] = q8KPattern(seeds[block], index);
            }
        }
        float[] vector = new float[blocks * QK_K];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 29 - 14) * 0.03125f;
        }

        float expected = referenceQ8K(scales, seeds, vector);
        float scalar = GgufQ8Dot.dotRowQ8PreparedWideScalar(blocks, QK_K, quants, scales, 0, 0, vector);
        float vectorized = GgufQ8Dot.dotRowQ8PreparedWideVector(blocks, QK_K, quants, scales, 0, 0, vector);

        assertEquals(expected, scalar, 1.0e-3f);
        assertEquals(expected, vectorized, 1.0e-3f);
    }

    private static void writePatternQ8KBlock(MemorySegment block, float scale, int seed) {
        writeQ8KBlock(block, scale, (byte) 0);
        for (int index = 0; index < QK_K; index++) {
            block.set(ValueLayout.JAVA_BYTE, Float.BYTES + index, q8KPattern(seed, index));
        }
    }

    private static float referenceRawQ8K(float[] scales, int[] seeds, float[] vector) {
        return referenceRawQ8K(scales, seeds, vector, 0);
    }

    private static float referenceRawQ8K(float[] scales, int[] seeds, float[] vector, int vectorOffset) {
        return referenceQ8K(scales, seeds, vector, vectorOffset);
    }

    private static float referenceQ8K(float[] scales, int[] seeds, float[] vector) {
        return referenceQ8K(scales, seeds, vector, 0);
    }

    private static float referenceQ8K(float[] scales, int[] seeds, float[] vector, int vectorOffset) {
        float sum = 0.0f;
        int vectorBase = vectorOffset;
        for (int block = 0; block < scales.length; block++) {
            float quantDot = 0.0f;
            for (int index = 0; index < QK_K; index++) {
                quantDot += q8KPattern(seeds[block], index) * vector[vectorBase + index];
            }
            sum += scales[block] * quantDot;
            vectorBase += QK_K;
        }
        return sum;
    }

    private static byte q8KPattern(int seed, int index) {
        return (byte) (seed + index * 37);
    }
}
