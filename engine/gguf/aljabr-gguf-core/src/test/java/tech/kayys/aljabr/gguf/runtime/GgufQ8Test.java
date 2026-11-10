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
import static org.junit.jupiter.api.Assertions.assertSame;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;

class GgufQ8Test {
    @Test
    void preparesAndCachesQ8MatrixForGenericMatVec() {
        String previous = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 34);
            writeQ8Block(segment.asSlice(0, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(34, 34), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8", new long[]{32, 2}, 8, 0, 2L * 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(32);

            assertEquals(32.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(64.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(64.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(32, first.quantStride());
            assertEquals(GgufTensorOps.Q8Matrix.ROW_ROUTE_32, first.rowRoute());
            assertEquals(expectedRowKernel(Q8_0_BLOCK_SIZE), first.rowKernel());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(32.0f, preparedOutput[0], 0.0f);
            assertEquals(64.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void preparedQ8VectorAccumulatorMatchesIndependentReferenceAcrossBlocks() {
        int blocks = 5;
        byte[] quants = new byte[blocks * Q8_0_BLOCK_SIZE];
        float[] scales = {1.0f, 2.0f, -1.0f, 0.5f, -1.75f};
        int[] seeds = {17, 43, 71, 103, 149};
        for (int block = 0; block < blocks; block++) {
            for (int index = 0; index < Q8_0_BLOCK_SIZE; index++) {
                quants[block * Q8_0_BLOCK_SIZE + index] = q8Pattern(seeds[block], index);
            }
        }
        float[] vector = new float[blocks * Q8_0_BLOCK_SIZE];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 23 - 11) * 0.0625f;
        }

        float expected = referencePreparedQ8(scales, seeds, vector);
        float scalar = GgufQ8Dot.dotRowQ8Prepared32Scalar(blocks, quants, scales, 0, 0, vector);
        float vectorized = GgufQ8Dot.dotRowQ8Prepared32Vector(blocks, quants, scales, 0, 0, vector);

        assertEquals(expected, scalar, 1.0e-3f);
        assertEquals(expected, vectorized, 1.0e-3f);
    }

    @Test
    void preparedQ8Vector16AccumulatorMatchesIndependentReferenceAcrossBlocks() {
        int blockSize = 16;
        int blocks = 5;
        byte[] quants = new byte[blocks * blockSize];
        float[] scales = {1.0f, -0.5f, 2.0f, -1.25f, 0.75f};
        int[] seeds = {19, 41, 73, 97, 131};
        for (int block = 0; block < blocks; block++) {
            for (int index = 0; index < blockSize; index++) {
                quants[block * blockSize + index] = q8Pattern(seeds[block], index);
            }
        }
        float[] vector = new float[blocks * blockSize];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 13 - 6) * 0.125f;
        }

        float expected = referencePreparedQ8(blockSize, scales, seeds, vector);
        float scalar = GgufQ8Dot.dotRowQ8Prepared16Scalar(blocks, quants, scales, 0, 0, vector);
        float vectorized = GgufQ8Dot.dotRowQ8Prepared16Vector(blocks, quants, scales, 0, 0, vector);

        assertEquals(expected, scalar, 1.0e-3f);
        assertEquals(expected, vectorized, 1.0e-3f);
    }

    @Test
    void preparedQ8GenericBlockVectorAccumulatorHandlesTailsAcrossBlocks() {
        int blockSize = 20;
        int blocks = 5;
        byte[] quants = new byte[blocks * blockSize];
        float[] scales = {0.75f, -1.5f, 2.25f, -0.625f, 1.375f};
        int[] seeds = {31, 59, 101, 137, 173};
        for (int block = 0; block < blocks; block++) {
            for (int index = 0; index < blockSize; index++) {
                quants[block * blockSize + index] = q8Pattern(seeds[block], index);
            }
        }
        float[] vector = new float[blocks * blockSize];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 11 - 5) * 0.1875f;
        }

        float expected = referencePreparedQ8(blockSize, scales, seeds, vector);
        float scalar = GgufQ8Dot.dotRowQ8PreparedBlockScalar(blocks, blockSize, quants, scales, 0, 0, vector);
        float vectorized = GgufQ8Dot.dotRowQ8PreparedBlockVector(blocks, blockSize, quants, scales, 0, 0, vector);

        assertEquals(expected, scalar, 1.0e-3f);
        assertEquals(expected, vectorized, 1.0e-3f);
    }

    @Test
    void preparedQ8RowWalkersHandleUnrolledRowsAndTailAcrossRoutes() {
        assertPreparedQ8RowsHandleUnrolledRowsAndTail(Q8_0_BLOCK_SIZE);
        assertPreparedQ8RowsHandleUnrolledRowsAndTail(16);
        assertPreparedQ8RowsHandleUnrolledRowsAndTail(128);
        assertPreparedQ8RowsHandleUnrolledRowsAndTail(20);
    }

    private static void assertPreparedQ8RowsHandleUnrolledRowsAndTail(int blockSize) {
        int rows = 7;
        int blocksPerRow = 5;
        int columns = blocksPerRow * blockSize;
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * blocksPerRow];
        for (int row = 0; row < rows; row++) {
            for (int block = 0; block < blocksPerRow; block++) {
                int matrixBlock = row * blocksPerRow + block;
                int seed = 23 + row * 31 + block * 17 + blockSize;
                scales[matrixBlock] = ((block & 1) == 0 ? 0.625f : -0.875f) + row * 0.03125f;
                int qBase = row * columns + block * blockSize;
                for (int index = 0; index < blockSize; index++) {
                    quants[qBase + index] = q8Pattern(seed, index);
                }
            }
        }
        float[] vector = new float[columns];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 29 - 14) * 0.0390625f;
        }
        GgufTensorOps.Q8Matrix matrix =
                new GgufTensorOps.Q8Matrix(columns, rows, blocksPerRow, blockSize, quants, scales);
        assertEquals(expectedRowRoute(blockSize), matrix.rowRoute());
        assertEquals(expectedRowKernel(blockSize), matrix.rowKernel());
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ8Rows.dotRowQ8(matrix, vector, row);
        }

        GgufQ8Rows.fillMatVecRowsQ8(matrix, vector, actual, 0, rows);

        assertArrayEquals(expected, actual, 0.0f);

        float[] expectedScalar = new float[rows];
        float[] scalar4 = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedScalar[row] = dotPreparedQ8Scalar(blockSize, blocksPerRow, quants, scales, row * blocksPerRow, row * columns, vector);
        }
        fillPreparedQ8Scalar4(blockSize, blocksPerRow, quants, scales, columns, vector, scalar4, rows);
        assertArrayEquals(expectedScalar, scalar4, 0.0f);
    }

    private static float dotPreparedQ8Scalar(
            int blockSize,
            int blocksPerRow,
            byte[] quants,
            float[] scales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        return switch (blockSize) {
            case Q8_0_BLOCK_SIZE -> GgufQ8Dot.dotRowQ8Prepared32Scalar(
                    blocksPerRow, quants, scales, matrixBlock, qBase, vector);
            case GgufQuantFormats.NVFP4_SUB_BLOCK_SIZE -> GgufQ8Dot.dotRowQ8Prepared16Scalar(
                    blocksPerRow, quants, scales, matrixBlock, qBase, vector);
            case GgufQuantFormats.Q1_0_BLOCK_SIZE, GgufQuantFormats.QK_K -> GgufQ8Dot.dotRowQ8PreparedWideScalar(
                    blocksPerRow, blockSize, quants, scales, matrixBlock, qBase, vector);
            default -> GgufQ8Dot.dotRowQ8PreparedBlockScalar(
                    blocksPerRow, blockSize, quants, scales, matrixBlock, qBase, vector);
        };
    }

    private static void fillPreparedQ8Scalar4(
            int blockSize,
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
            dotPreparedQ8Scalar4(blockSize, blocksPerRow, quants, scales, row * blocksPerRow, row * qStride, qStride, vector, output, row);
        }
        for (; row < rows; row++) {
            output[row] = dotPreparedQ8Scalar(
                    blockSize, blocksPerRow, quants, scales, row * blocksPerRow, row * qStride, vector);
        }
    }

    private static void dotPreparedQ8Scalar4(
            int blockSize,
            int blocksPerRow,
            byte[] quants,
            float[] scales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        switch (blockSize) {
            case Q8_0_BLOCK_SIZE ->
                    GgufQ8Dot.dotRowsQ8Prepared32Scalar4(
                            blocksPerRow, quants, scales, matrixBlock, qBase, qStride, vector, output, outputOffset);
            case GgufQuantFormats.NVFP4_SUB_BLOCK_SIZE ->
                    GgufQ8Dot.dotRowsQ8Prepared16Scalar4(
                            blocksPerRow, quants, scales, matrixBlock, qBase, qStride, vector, output, outputOffset);
            case GgufQuantFormats.Q1_0_BLOCK_SIZE, GgufQuantFormats.QK_K ->
                    GgufQ8Dot.dotRowsQ8PreparedWideScalar4(
                            blocksPerRow, blockSize, quants, scales, matrixBlock, qBase, qStride, vector, output, outputOffset);
            default ->
                    GgufQ8Dot.dotRowsQ8PreparedBlockScalar4(
                            blocksPerRow, blockSize, quants, scales, matrixBlock, qBase, qStride, vector, output, outputOffset);
        }
    }

    private static float referencePreparedQ8(float[] scales, int[] seeds, float[] vector) {
        return referencePreparedQ8(Q8_0_BLOCK_SIZE, scales, seeds, vector);
    }

    private static float referencePreparedQ8(int blockSize, float[] scales, int[] seeds, float[] vector) {
        float sum = 0.0f;
        int vectorBase = 0;
        for (int block = 0; block < scales.length; block++) {
            for (int index = 0; index < blockSize; index++) {
                sum += scales[block] * q8Pattern(seeds[block], index) * vector[vectorBase + index];
            }
            vectorBase += blockSize;
        }
        return sum;
    }

    private static byte q8Pattern(int seed, int index) {
        return (byte) (seed + index * 37);
    }

    private static int expectedRowRoute(int blockSize) {
        return switch (blockSize) {
            case GgufQuantFormats.Q4_0_BLOCK_SIZE -> GgufTensorOps.Q8Matrix.ROW_ROUTE_32;
            case GgufQuantFormats.NVFP4_SUB_BLOCK_SIZE -> GgufTensorOps.Q8Matrix.ROW_ROUTE_16;
            case GgufQuantFormats.Q1_0_BLOCK_SIZE, GgufQuantFormats.QK_K ->
                    GgufTensorOps.Q8Matrix.ROW_ROUTE_WIDE;
            default -> GgufTensorOps.Q8Matrix.ROW_ROUTE_BLOCK;
        };
    }

    private static int expectedRowKernel(int blockSize) {
        return switch (expectedRowRoute(blockSize)) {
            case GgufTensorOps.Q8Matrix.ROW_ROUTE_32 -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? GgufTensorOps.Q8Matrix.ROW_KERNEL_32_VECTOR
                    : GgufTensorOps.Q8Matrix.ROW_KERNEL_32_SCALAR;
            case GgufTensorOps.Q8Matrix.ROW_ROUTE_16 -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? GgufTensorOps.Q8Matrix.ROW_KERNEL_16_VECTOR
                    : GgufTensorOps.Q8Matrix.ROW_KERNEL_16_SCALAR;
            case GgufTensorOps.Q8Matrix.ROW_ROUTE_WIDE -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? GgufTensorOps.Q8Matrix.ROW_KERNEL_WIDE_VECTOR
                    : GgufTensorOps.Q8Matrix.ROW_KERNEL_WIDE_SCALAR;
            default -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? GgufTensorOps.Q8Matrix.ROW_KERNEL_BLOCK_VECTOR
                    : GgufTensorOps.Q8Matrix.ROW_KERNEL_BLOCK_SCALAR;
        };
    }
}
