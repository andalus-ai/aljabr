package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_1_BLOCK_BYTES;

class GgufQ32PrepBiasTest {
    @Test
    void preparesQ32MatrixForQ5_1HighBitsAndBias() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(Q5_1_BLOCK_BYTES);
            writeQ5_1Block(segment, (short) 0x3c00, (short) 0x3800, -1, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_1", new long[]{32, 1}, 7, 0, Q5_1_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q32Matrix matrix = GgufTensorOps.q32Matrix(model, tensor);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(matrix, ones(32), output, 1, true);

            assertEquals(32, matrix.quantStride());
            assertTrue(matrix.hasBlockBiases());
            assertEquals(1, matrix.blockBiases().length);
            assertEquals(expectedPrecomputedBiasKernel(), matrix.precomputedBiasKernel());
            assertEquals(expectedDirectBiasKernel(), matrix.directBiasKernel());
            assertEquals(40L, matrix.estimatedBytes());
            assertEquals(576.0f, output[0], 0.0f);
        }
    }

    @Test
    void preparedQ32SingleRowBiasMatVecMatchesDotRow() {
        try (Arena arena = Arena.ofShared()) {
            long q5Offset = Q4_1_BLOCK_BYTES;
            MemorySegment segment = arena.allocate(Q4_1_BLOCK_BYTES + Q5_1_BLOCK_BYTES);
            writeQ4_1Block(segment.asSlice(0, Q4_1_BLOCK_BYTES), (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            writeQ5_1Block(
                    segment.asSlice(q5Offset, Q5_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    -1,
                    (byte) 0x21);
            GGUFTensorInfo q4Tensor =
                    new GGUFTensorInfo("q4_1.prepared.single", new long[]{32, 1}, 3, 0, Q4_1_BLOCK_BYTES);
            GGUFTensorInfo q5Tensor =
                    new GGUFTensorInfo("q5_1.prepared.single", new long[]{32, 1}, 7, q5Offset, Q5_1_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q4Tensor, q5Tensor), 0, segment, null);

            float[] vector = ramp(32);
            float[] q4Output = new float[1];
            float[] q5Output = new float[1];
            GgufTensorOps.matVecRows(GgufTensorOps.q32Matrix(model, q4Tensor), vector, q4Output, 1, true);
            GgufTensorOps.matVecRows(GgufTensorOps.q32Matrix(model, q5Tensor), vector, q5Output, 1, true);

            assertEquals(GgufTensorOps.dotRow(model, q4Tensor, 0, vector), q4Output[0], 0.0f);
            assertEquals(GgufTensorOps.dotRow(model, q5Tensor, 0, vector), q5Output[0], 0.0f);
        }
    }

    @Test
    void preparedQ32DirectBiasDotMatchesPrecomputedVectorSumsAcrossBlocks() {
        int columns = Q4_0_BLOCK_SIZE * 2;
        long q5Offset = 2L * Q4_1_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(q5Offset + 2L * Q5_1_BLOCK_BYTES);
            writeQ4_1Block(segment.asSlice(0, Q4_1_BLOCK_BYTES), (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            writeQ4_1Block(
                    segment.asSlice(Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                    (short) 0x4000,
                    (short) 0x3400,
                    (byte) 0x43);
            writeQ5_1Block(
                    segment.asSlice(q5Offset, Q5_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    0x5A5A_A5A5,
                    (byte) 0x21);
            writeQ5_1Block(
                    segment.asSlice(q5Offset + Q5_1_BLOCK_BYTES, Q5_1_BLOCK_BYTES),
                    (short) 0x4000,
                    (short) 0x3400,
                    0xA5A5_5A5A,
                    (byte) 0x43);
            GGUFTensorInfo q4Tensor =
                    new GGUFTensorInfo("q4_1.prepared.direct.multi_block", new long[]{columns, 1}, 3, 0, q5Offset);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo(
                    "q5_1.prepared.direct.multi_block",
                    new long[]{columns, 1},
                    7,
                    q5Offset,
                    2L * Q5_1_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q4Tensor, q5Tensor), 0, segment, null);
            float[] vector = fractionalRamp(columns);
            float[] vectorGroupSums =
                    GgufSum.vector32GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
            GgufTensorOps.Q32Matrix q4Matrix = GgufTensorOps.q32Matrix(model, q4Tensor);
            GgufTensorOps.Q32Matrix q5Matrix = GgufTensorOps.q32Matrix(model, q5Tensor);

            assertEquals(
                    GgufQ32Rows.dotRowQ32(q4Matrix, vector, vectorGroupSums, 0),
                    GgufQ32Rows.dotRowQ32Direct(q4Matrix, vector, 0),
                    1.0e-4f);
            assertEquals(
                    GgufQ32Rows.dotRowQ32(q5Matrix, vector, vectorGroupSums, 0),
                    GgufQ32Rows.dotRowQ32Direct(q5Matrix, vector, 0),
                    1.0e-4f);
        }
    }

    @Test
    void preparedQ32DirectBiasVectorAccumulatorMatchesIndependentReferenceAcrossBlocks() {
        int blocks = 7;
        byte[] quants = new byte[blocks * Q4_0_BLOCK_SIZE];
        float[] scales = {1.0f, -0.75f, 2.25f, 0.5f, -1.5f, 0.875f, -0.3125f};
        float[] biases = {0.5f, -1.25f, 0.875f, -0.375f, 1.125f, -0.625f, 1.75f};
        int[] seeds = {31, 73, 127, 167, 211, 23, 97};
        for (int block = 0; block < blocks; block++) {
            for (int index = 0; index < Q4_0_BLOCK_SIZE; index++) {
                quants[block * Q4_0_BLOCK_SIZE + index] = q32Pattern(seeds[block], index);
            }
        }
        float[] vector = new float[blocks * Q4_0_BLOCK_SIZE];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 19 - 9) * 0.078125f;
        }
        float[] vectorSums = groupSums(Q4_0_BLOCK_SIZE, blocks, vector);

        float expected = referenceQ32Bias(scales, biases, seeds, vector);
        float scalar = GgufQ32Dot.dotRowQ32PreparedDirectScalar(blocks, quants, scales, biases, 0, 0, vector);
        float precomputedScalar = GgufQ32Dot.dotRowQ32PreparedScalar(
                blocks, quants, scales, biases, 0, 0, vector, vectorSums);
        float precomputed = GgufQ32Dot.dotRowQ32PreparedVector(blocks, quants, scales, biases, 0, 0, vector, vectorSums);
        float vectorized = GgufQ32Dot.dotRowQ32PreparedDirectVector(blocks, quants, scales, biases, 0, 0, vector);

        assertEquals(expected, scalar, 1.0e-3f);
        assertEquals(expected, precomputedScalar, 1.0e-3f);
        assertEquals(expected, precomputed, 1.0e-3f);
        assertEquals(expected, vectorized, 1.0e-3f);
    }

    @Test
    void preparedQ32BiasRowWalkersHandleUnrolledRowsAndTail() {
        int rows = 7;
        int blocksPerRow = 5;
        int columns = blocksPerRow * Q4_0_BLOCK_SIZE;
        byte[] quants = new byte[rows * blocksPerRow * Q4_0_BLOCK_SIZE];
        float[] scales = new float[rows * blocksPerRow];
        float[] biases = new float[rows * blocksPerRow];
        for (int row = 0; row < rows; row++) {
            for (int block = 0; block < blocksPerRow; block++) {
                int matrixBlock = row * blocksPerRow + block;
                int seed = 23 + row * 31 + block * 47;
                scales[matrixBlock] = ((row & 1) == 0 ? 1.125f : -0.6875f) + block * 0.15625f;
                biases[matrixBlock] = ((block & 1) == 0 ? 0.375f : -0.5f) + row * 0.03125f;
                for (int index = 0; index < Q4_0_BLOCK_SIZE; index++) {
                    quants[matrixBlock * Q4_0_BLOCK_SIZE + index] = q32Pattern(seed, index);
                }
            }
        }
        float[] vector = new float[columns];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 29 - 14) * 0.0390625f;
        }
        float[] vectorGroupSums =
                GgufSum.vector32GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
        GgufTensorOps.Q32Matrix matrix = new GgufTensorOps.Q32Matrix(
                columns, rows, blocksPerRow, quants, scales, biases, true);
        assertEquals(expectedPrecomputedBiasKernel(), matrix.precomputedBiasKernel());
        assertEquals(expectedDirectBiasKernel(), matrix.directBiasKernel());
        float[] expectedPrecomputed = new float[rows];
        float[] actualPrecomputed = new float[rows];
        float[] expectedDirect = new float[rows];
        float[] actualDirect = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedPrecomputed[row] = GgufQ32Rows.dotRowQ32(matrix, vector, vectorGroupSums, row);
            expectedDirect[row] = GgufQ32Rows.dotRowQ32Direct(matrix, vector, row);
        }

        GgufQ32Rows.fillMatVecRowsQ32(matrix, vector, vectorGroupSums, actualPrecomputed, 0, rows);
        GgufQ32Rows.fillMatVecRowsQ32Direct(matrix, vector, actualDirect, 0, rows);

        assertArrayEquals(expectedPrecomputed, actualPrecomputed, 0.0f);
        assertArrayEquals(expectedDirect, actualDirect, 0.0f);

        float[] expectedPrecomputedScalar = new float[rows];
        float[] actualPrecomputedScalar = new float[rows];
        float[] expectedDirectScalar = new float[rows];
        float[] actualDirectScalar = new float[rows];
        for (int row = 0; row < rows; row++) {
            int matrixBlock = row * blocksPerRow;
            int qBase = row * columns;
            expectedPrecomputedScalar[row] = GgufQ32Dot.dotRowQ32PreparedScalar(
                    blocksPerRow, quants, scales, biases, matrixBlock, qBase, vector, vectorGroupSums);
            expectedDirectScalar[row] = GgufQ32Dot.dotRowQ32PreparedDirectScalar(
                    blocksPerRow, quants, scales, biases, matrixBlock, qBase, vector);
        }
        fillPrecomputedScalar4(
                blocksPerRow, quants, scales, biases, columns, vector, vectorGroupSums, actualPrecomputedScalar, rows);
        fillDirectScalar4(blocksPerRow, quants, scales, biases, columns, vector, actualDirectScalar, rows);
        assertArrayEquals(expectedPrecomputedScalar, actualPrecomputedScalar, 0.0f);
        assertArrayEquals(expectedDirectScalar, actualDirectScalar, 0.0f);
    }

    private static void fillPrecomputedScalar4(
            int blocksPerRow,
            byte[] quants,
            float[] scales,
            float[] biases,
            int qStride,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufQ32Dot.dotRowsQ32PreparedScalar4(
                    blocksPerRow,
                    quants,
                    scales,
                    biases,
                    row * blocksPerRow,
                    blocksPerRow,
                    row * qStride,
                    qStride,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufQ32Dot.dotRowQ32PreparedScalar(
                    blocksPerRow, quants, scales, biases, row * blocksPerRow, row * qStride, vector, vectorGroupSums);
        }
    }

    private static void fillDirectScalar4(
            int blocksPerRow,
            byte[] quants,
            float[] scales,
            float[] biases,
            int qStride,
            float[] vector,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufQ32Dot.dotRowsQ32PreparedDirectScalar4(
                    blocksPerRow,
                    quants,
                    scales,
                    biases,
                    row * blocksPerRow,
                    blocksPerRow,
                    row * qStride,
                    qStride,
                    vector,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufQ32Dot.dotRowQ32PreparedDirectScalar(
                    blocksPerRow, quants, scales, biases, row * blocksPerRow, row * qStride, vector);
        }
    }

    private static float[] fractionalRamp(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i + 1) * 0.125f + ((i & 1) == 0 ? 0.5f : -0.25f);
        }
        return values;
    }

    private static float referenceQ32Bias(float[] scales, float[] biases, int[] seeds, float[] vector) {
        float sum = 0.0f;
        int vectorBase = 0;
        for (int block = 0; block < scales.length; block++) {
            for (int index = 0; index < Q4_0_BLOCK_SIZE; index++) {
                sum += (scales[block] * q32Pattern(seeds[block], index) + biases[block])
                        * vector[vectorBase + index];
            }
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static float[] groupSums(int blockSize, int blocks, float[] vector) {
        float[] sums = new float[blocks];
        int vectorBase = 0;
        for (int block = 0; block < blocks; block++) {
            float sum = 0.0f;
            for (int index = 0; index < blockSize; index++) {
                sum += vector[vectorBase + index];
            }
            sums[block] = sum;
            vectorBase += blockSize;
        }
        return sums;
    }

    private static byte q32Pattern(int seed, int index) {
        return (byte) (seed + index * 41);
    }

    private static int expectedPrecomputedBiasKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q32Matrix.ROW_KERNEL_PRECOMPUTED_BIAS_VECTOR
                : GgufTensorOps.Q32Matrix.ROW_KERNEL_PRECOMPUTED_BIAS_SCALAR;
    }

    private static int expectedDirectBiasKernel() {
        return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                ? GgufTensorOps.Q32Matrix.ROW_KERNEL_DIRECT_BIAS_VECTOR
                : GgufTensorOps.Q32Matrix.ROW_KERNEL_DIRECT_BIAS_SCALAR;
    }
}
