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
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;

class GgufQ32PrepTest {
    @Test
    void preparesAndCachesQ32MatrixForGenericMatVec() {
        String previous = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4_0", new long[]{32, 2}, 2, 0, 2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(32);
            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(16.0f, output[0], 0.0f);
            assertEquals(48.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.q32MatrixCacheBytes(model));

            GgufTensorOps.Q32Matrix first = GgufTensorOps.q32MatrixCached(model, tensor);
            GgufTensorOps.Q32Matrix second = GgufTensorOps.q32MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(32, first.quantStride());
            assertFalse(first.hasBlockBiases());
            assertEquals(0, first.blockBiases().length);
            assertEquals(expectedNoBiasKernel(), first.noBiasKernel());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(16.0f, preparedOutput[0], 0.0f);
            assertEquals(48.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ32MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previous);
        }
    }

    @Test
    void preparedQ32NoBiasVectorAccumulatorMatchesIndependentReferenceAcrossBlocks() {
        int blocks = 5;
        byte[] quants = new byte[blocks * Q4_0_BLOCK_SIZE];
        float[] scales = {1.0f, -0.75f, 2.25f, 0.5f, -1.5f};
        int[] seeds = {23, 61, 109, 151, 197};
        for (int block = 0; block < blocks; block++) {
            for (int index = 0; index < Q4_0_BLOCK_SIZE; index++) {
                quants[block * Q4_0_BLOCK_SIZE + index] = q32Pattern(seeds[block], index);
            }
        }
        float[] vector = new float[blocks * Q4_0_BLOCK_SIZE];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 17 - 8) * 0.09375f;
        }

        float expected = referenceQ32NoBias(scales, seeds, vector);
        float scalar = GgufQ32Dot.dotRowQ32PreparedNoBiasScalar(blocks, quants, scales, 0, 0, vector);
        float vectorized = GgufQ32Dot.dotRowQ32PreparedNoBiasVector(blocks, quants, scales, 0, 0, vector);

        assertEquals(expected, scalar, 1.0e-3f);
        assertEquals(expected, vectorized, 1.0e-3f);
    }

    @Test
    void preparedQ32NoBiasRowWalkerHandlesUnrolledRowsAndTail() {
        int rows = 7;
        int blocksPerRow = 5;
        int columns = blocksPerRow * Q4_0_BLOCK_SIZE;
        byte[] quants = new byte[rows * blocksPerRow * Q4_0_BLOCK_SIZE];
        float[] scales = new float[rows * blocksPerRow];
        for (int row = 0; row < rows; row++) {
            for (int block = 0; block < blocksPerRow; block++) {
                int matrixBlock = row * blocksPerRow + block;
                int seed = 17 + row * 29 + block * 53;
                scales[matrixBlock] = ((row & 1) == 0 ? 0.875f : -0.625f) + block * 0.1875f;
                for (int index = 0; index < Q4_0_BLOCK_SIZE; index++) {
                    quants[matrixBlock * Q4_0_BLOCK_SIZE + index] = q32Pattern(seed, index);
                }
            }
        }
        float[] vector = new float[columns];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 23 - 11) * 0.046875f;
        }
        GgufTensorOps.Q32Matrix matrix = new GgufTensorOps.Q32Matrix(
                columns, rows, blocksPerRow, quants, scales, new float[0], false);
        assertEquals(expectedNoBiasKernel(), matrix.noBiasKernel());
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ32Rows.dotRowQ32(matrix, vector, null, row);
        }

        GgufQ32Rows.fillMatVecRowsQ32(matrix, vector, null, actual, 0, rows);

        assertArrayEquals(expected, actual, 0.0f);

        float[] expectedScalar = new float[rows];
        float[] scalar4 = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedScalar[row] = GgufQ32Dot.dotRowQ32PreparedNoBiasScalar(
                    blocksPerRow, quants, scales, row * blocksPerRow, row * columns, vector);
        }
        fillNoBiasScalar4(blocksPerRow, quants, scales, columns, vector, scalar4, rows);
        assertArrayEquals(expectedScalar, scalar4, 0.0f);
    }

    private static void fillNoBiasScalar4(
            int blocksPerRow,
            byte[] quants,
            float[] scales,
            int qStride,
            float[] vector,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufQ32Dot.dotRowsQ32PreparedNoBiasScalar4(
                    blocksPerRow,
                    quants,
                    scales,
                    row * blocksPerRow,
                    blocksPerRow,
                    row * qStride,
                    qStride,
                    vector,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufQ32Dot.dotRowQ32PreparedNoBiasScalar(
                    blocksPerRow, quants, scales, row * blocksPerRow, row * qStride, vector);
        }
    }

    private static float referenceQ32NoBias(float[] scales, int[] seeds, float[] vector) {
        float sum = 0.0f;
        int vectorBase = 0;
        for (int block = 0; block < scales.length; block++) {
            for (int index = 0; index < Q4_0_BLOCK_SIZE; index++) {
                sum += scales[block] * q32Pattern(seeds[block], index) * vector[vectorBase + index];
            }
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static byte q32Pattern(int seed, int index) {
        return (byte) (seed + index * 43);
    }

    private static int expectedNoBiasKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q32Matrix.ROW_KERNEL_NO_BIAS_VECTOR
                : GgufTensorOps.Q32Matrix.ROW_KERNEL_NO_BIAS_SCALAR;
    }
}
