package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ3KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KNoMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KNoMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ6KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class GgufKPrepTest {
    @Test
    void preparedSingleRowNoMinMatVecMatchesDotRowAcrossFamilies() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(776);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 1, (byte) 0x55);
            writeQ3KBlock(segment.asSlice(84, 110), 1, (byte) 0x55, (byte) 0xFF);
            writeQ4KNoMinLaneOrderBlock(segment.asSlice(194, 144));
            writeQ5KNoMinLaneOrderBlock(segment.asSlice(338, 176));
            writeQ6KBlock(segment.asSlice(514, 210), (byte) 0x11, (byte) 0xAA, (byte) 1);
            writeQ8Block(segment.asSlice(724, 34), (short) 0x3c00, (byte) 2);
            writeQ4_0Block(segment.asSlice(758, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo q2Tensor = new GGUFTensorInfo("q2.prepared.single.no_mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q3Tensor = new GGUFTensorInfo("q3.prepared.single", new long[]{256, 1}, 11, 84, 110);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4.prepared.single.no_mins", new long[]{256, 1}, 12, 194, 144);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5.prepared.single.no_mins", new long[]{256, 1}, 13, 338, 176);
            GGUFTensorInfo q6Tensor = new GGUFTensorInfo("q6.prepared.single", new long[]{256, 1}, 14, 514, 210);
            GGUFTensorInfo q8Tensor = new GGUFTensorInfo("q8.prepared.single", new long[]{32, 1}, 8, 724, 34);
            GGUFTensorInfo q32Tensor = new GGUFTensorInfo("q32.prepared.single", new long[]{32, 1}, 2, 758, 18);
            GGUFModel model = new GGUFModel(
                    3,
                    Map.of(),
                    List.of(q2Tensor, q3Tensor, q4Tensor, q5Tensor, q6Tensor, q8Tensor, q32Tensor),
                    0,
                    segment,
                    null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.Q2KMatrix q2Matrix = GgufTensorOps.q2KMatrix(model, q2Tensor);
            GgufTensorOps.Q4KMatrix q4Matrix = GgufTensorOps.q4KMatrix(model, q4Tensor);
            GgufTensorOps.Q5KMatrix q5Matrix = GgufTensorOps.q5KMatrix(model, q5Tensor);

            assertFalse(q2Matrix.hasGroupMins());
            assertFalse(q4Matrix.hasGroupMins());
            assertFalse(q5Matrix.hasGroupMins());

            GgufTensorOps.matVecRows(q2Matrix, vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q2Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(GgufTensorOps.q3KMatrix(model, q3Tensor), vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q3Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(q4Matrix, vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q4Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(q5Matrix, vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q5Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(GgufTensorOps.q6KMatrix(model, q6Tensor), vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q6Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(GgufTensorOps.q8Matrix(model, q8Tensor), vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q8Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(GgufTensorOps.q32Matrix(model, q32Tensor), vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q32Tensor, 0, vector), output[0], 0.0f);
        }
    }

    @Test
    void preparedTinyNoMinMatVecRowsMatchDirectWalkersAcrossFamilies() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        try {
            assertTinyK16NoMinRows();
            assertTinyK32NoMinRows();
            assertTinyQ32NoBiasRows();
            assertTinyQ8Rows();
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
        }
    }

    @Test
    void preparedKNoMinVectorAccumulatorsMatchIndependentReferences() {
        int groups16 = 5;
        byte[] quants16 = new byte[groups16 * 16];
        float[] scales16 = {1.0f, -0.5f, 2.0f, -1.25f, 0.75f};
        int[] seeds16 = {17, 47, 83, 127, 173};
        fillPatternedQuants(quants16, 16, seeds16);
        float[] vector16 = patternedVector(groups16 * 16, 19, 0.0625f);

        float expected16 = referenceNoMin(16, scales16, seeds16, vector16);
        assertEquals(
                expected16,
                GgufKDot.dotRowK16PreparedNoMinsScalar(groups16, quants16, scales16, 0, 0, vector16),
                1.0e-3f);
        assertEquals(
                expected16,
                GgufKDot.dotRowK16PreparedNoMinsVector(groups16, quants16, scales16, 0, 0, vector16),
                1.0e-3f);

        int groups32 = 5;
        byte[] quants32 = new byte[groups32 * 32];
        float[] scales32 = {0.75f, -1.5f, 2.25f, -0.625f, 1.375f};
        int[] seeds32 = {29, 71, 131, 181, 223};
        fillPatternedQuants(quants32, 32, seeds32);
        float[] vector32 = patternedVector(groups32 * 32, 23, 0.09375f);

        float expected32 = referenceNoMin(32, scales32, seeds32, vector32);
        assertEquals(
                expected32,
                GgufKDot.dotRowK32GroupsPreparedNoMinsScalar(groups32, quants32, scales32, 0, 0, vector32),
                1.0e-3f);
        assertEquals(
                expected32,
                GgufKDot.dotRowK32GroupsPreparedNoMinsVector(groups32, quants32, scales32, 0, 0, vector32),
                1.0e-3f);
    }

    @Test
    void preparedK32RowWalkerHandlesUnrolledRowsAndTail() {
        int rows = 7;
        int blocksPerRow = 2;
        int columns = blocksPerRow * QK_K;
        int groupsPerRow = blocksPerRow * (QK_K / 32);
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * groupsPerRow];
        float[] mins = new float[rows * groupsPerRow];
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groupsPerRow; group++) {
                int groupBase = row * groupsPerRow + group;
                int seed = 19 + row * 31 + group * 17;
                scales[groupBase] = ((group & 1) == 0 ? 0.75f : -1.125f) + row * 0.03125f;
                mins[groupBase] = ((row & 1) == 0 ? 0.25f : -0.375f) + group * 0.015625f;
                int quantBase = row * columns + group * 32;
                for (int index = 0; index < 32; index++) {
                    quants[quantBase + index] = kPattern(seed, index);
                }
            }
        }
        float[] vector = patternedVector(columns, 31, 0.03125f);
        float[] vectorGroupSums =
                GgufSum.q4KVectorGroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
        float[] expectedNoMin = new float[rows];
        float[] actualNoMin = new float[rows];
        float[] expectedWithMins = new float[rows];
        float[] actualWithMins = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedNoMin[row] =
                    GgufKRows.dotRowK32Prepared(blocksPerRow, quants, scales, new float[0], vector, null, row);
            expectedWithMins[row] =
                    GgufKRows.dotRowK32Prepared(blocksPerRow, quants, scales, mins, vector, vectorGroupSums, row);
        }

        GgufKRows.fillMatVecRowsK32Prepared(
                blocksPerRow, quants, scales, new float[0], vector, null, actualNoMin, 0, rows);
        GgufKRows.fillMatVecRowsK32Prepared(
                blocksPerRow, quants, scales, mins, vector, vectorGroupSums, actualWithMins, 0, rows);

        assertArrayEquals(expectedNoMin, actualNoMin, 0.0f);
        assertArrayEquals(expectedWithMins, actualWithMins, 0.0f);

        float[] expectedNoMinScalar = new float[rows];
        float[] actualNoMinScalar = new float[rows];
        float[] actualNoMinKernelVector = new float[rows];
        float[] actualNoMinKernelScalar = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedNoMinScalar[row] = GgufKDot.dotRowK32GroupsPreparedNoMinsScalar(
                    groupsPerRow, quants, scales, row * columns, row * groupsPerRow, vector);
        }
        fillK32NoMinScalar4(groupsPerRow, quants, scales, columns, vector, actualNoMinScalar, rows);
        GgufKRows.fillMatVecRowsK32Prepared(
                GgufTensorOps.Q4KMatrix.ROW_KERNEL_NO_MIN_VECTOR,
                columns,
                groupsPerRow,
                quants,
                scales,
                new float[0],
                vector,
                null,
                actualNoMinKernelVector,
                0,
                rows);
        GgufKRows.fillMatVecRowsK32Prepared(
                GgufTensorOps.Q4KMatrix.ROW_KERNEL_NO_MIN_SCALAR,
                columns,
                groupsPerRow,
                quants,
                scales,
                new float[0],
                vector,
                null,
                actualNoMinKernelScalar,
                0,
                rows);
        assertArrayEquals(expectedNoMinScalar, actualNoMinScalar, 0.0f);
        assertArrayEquals(expectedNoMinScalar, actualNoMinKernelVector, 1.0e-3f);
        assertArrayEquals(expectedNoMinScalar, actualNoMinKernelScalar, 0.0f);
        assertEquals(
                expectedNoMinScalar[rows - 1],
                GgufKRows.dotRowK32Prepared(
                        GgufTensorOps.Q5KMatrix.ROW_KERNEL_NO_MIN_VECTOR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        new float[0],
                        vector,
                        null,
                        rows - 1),
                1.0e-3f);
        assertEquals(
                expectedNoMinScalar[rows - 1],
                GgufKRows.dotRowK32Prepared(
                        GgufTensorOps.Q5KMatrix.ROW_KERNEL_NO_MIN_SCALAR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        new float[0],
                        vector,
                        null,
                        rows - 1),
                0.0f);
    }

    @Test
    void preparedK16NoMinRowWalkerHandlesUnrolledRowsAndTail() {
        int rows = 7;
        int blocksPerRow = 2;
        int columns = blocksPerRow * QK_K;
        int groupsPerRow = blocksPerRow * (QK_K / 16);
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * groupsPerRow];
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groupsPerRow; group++) {
                int groupBase = row * groupsPerRow + group;
                int seed = 13 + row * 41 + group * 11;
                scales[groupBase] = ((group & 1) == 0 ? 0.625f : -0.8125f) + row * 0.0234375f;
                int quantBase = row * columns + group * 16;
                for (int index = 0; index < 16; index++) {
                    quants[quantBase + index] = kPattern(seed, index);
                }
            }
        }
        float[] vector = patternedVector(columns, 37, 0.02734375f);
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufKRows.dotRowK16PreparedNoMins(blocksPerRow, quants, scales, vector, row);
        }

        GgufKRows.fillMatVecRowsK16PreparedNoMins(blocksPerRow, quants, scales, vector, actual, 0, rows);

        assertArrayEquals(expected, actual, 0.0f);

        float[] expectedScalar = new float[rows];
        float[] actualScalar = new float[rows];
        float[] actualKernelVector = new float[rows];
        float[] actualKernelScalar = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedScalar[row] = GgufKDot.dotRowK16PreparedNoMinsScalar(
                    groupsPerRow, quants, scales, row * columns, row * groupsPerRow, vector);
        }
        fillK16NoMinScalar4(groupsPerRow, quants, scales, columns, vector, actualScalar, rows);
        GgufKRows.fillMatVecRowsK16PreparedNoMins(
                GgufTensorOps.Q3KMatrix.ROW_KERNEL_VECTOR,
                columns,
                groupsPerRow,
                quants,
                scales,
                vector,
                actualKernelVector,
                0,
                rows);
        GgufKRows.fillMatVecRowsK16PreparedNoMins(
                GgufTensorOps.Q3KMatrix.ROW_KERNEL_SCALAR,
                columns,
                groupsPerRow,
                quants,
                scales,
                vector,
                actualKernelScalar,
                0,
                rows);
        assertArrayEquals(expectedScalar, actualScalar, 0.0f);
        assertArrayEquals(expectedScalar, actualKernelVector, 1.0e-3f);
        assertArrayEquals(expectedScalar, actualKernelScalar, 0.0f);
        assertEquals(
                expectedScalar[rows - 1],
                GgufKRows.dotRowK16PreparedNoMins(
                        GgufTensorOps.Q6KMatrix.ROW_KERNEL_VECTOR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        vector,
                        rows - 1),
                1.0e-3f);
        assertEquals(
                expectedScalar[rows - 1],
                GgufKRows.dotRowK16PreparedNoMins(
                        GgufTensorOps.Q6KMatrix.ROW_KERNEL_SCALAR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        vector,
                        rows - 1),
                0.0f);
    }

    private static void fillK32NoMinScalar4(
            int groupsPerRow,
            byte[] quants,
            float[] scales,
            int rowQuantStride,
            float[] vector,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufKDot.dotRowsK32GroupsPreparedNoMinsScalar4(
                    groupsPerRow,
                    quants,
                    scales,
                    row * rowQuantStride,
                    rowQuantStride,
                    row * groupsPerRow,
                    groupsPerRow,
                    vector,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufKDot.dotRowK32GroupsPreparedNoMinsScalar(
                    groupsPerRow, quants, scales, row * rowQuantStride, row * groupsPerRow, vector);
        }
    }

    private static void fillK16NoMinScalar4(
            int groupsPerRow,
            byte[] quants,
            float[] scales,
            int rowQuantStride,
            float[] vector,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufKDot.dotRowsK16PreparedNoMinsScalar4(
                    groupsPerRow,
                    quants,
                    scales,
                    row * rowQuantStride,
                    rowQuantStride,
                    row * groupsPerRow,
                    groupsPerRow,
                    vector,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufKDot.dotRowK16PreparedNoMinsScalar(
                    groupsPerRow, quants, scales, row * rowQuantStride, row * groupsPerRow, vector);
        }
    }

    private static void assertTinyK16NoMinRows() {
        int rows = 4;
        int blocksPerRow = 1;
        int columns = blocksPerRow * QK_K;
        int groupsPerRow = blocksPerRow * (QK_K / 16);
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * groupsPerRow];
        fillPreparedRows(quants, scales, rows, columns, groupsPerRow, 16);
        float[] vector = patternedVector(columns, 41, 0.0234375f);
        float[] expected = new float[rows];

        GgufKRows.fillMatVecRowsK16PreparedNoMins(
                columns, groupsPerRow, quants, scales, vector, expected, 0, rows);

        float[] q2 = new float[rows];
        float[] q3 = new float[rows];
        float[] q6 = new float[rows];
        GgufTensorOps.matVecRows(
                new GgufTensorOps.Q2KMatrix(columns, rows, blocksPerRow, quants, scales, new float[0], false),
                vector,
                q2,
                rows,
                true);
        GgufTensorOps.matVecRows(
                new GgufTensorOps.Q3KMatrix(columns, rows, blocksPerRow, quants, scales),
                vector,
                q3,
                rows,
                true);
        GgufTensorOps.matVecRows(
                new GgufTensorOps.Q6KMatrix(columns, rows, blocksPerRow, quants, scales),
                vector,
                q6,
                rows,
                true);

        assertArrayEquals(expected, q2, 0.0f);
        assertArrayEquals(expected, q3, 0.0f);
        assertArrayEquals(expected, q6, 0.0f);
    }

    private static void assertTinyK32NoMinRows() {
        int rows = 4;
        int blocksPerRow = 1;
        int columns = blocksPerRow * QK_K;
        int groupsPerRow = blocksPerRow * (QK_K / 32);
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * groupsPerRow];
        float[] emptyMins = new float[0];
        fillPreparedRows(quants, scales, rows, columns, groupsPerRow, 32);
        float[] vector = patternedVector(columns, 43, 0.01953125f);
        float[] expected = new float[rows];

        GgufKRows.fillMatVecRowsK32Prepared(
                columns, groupsPerRow, quants, scales, emptyMins, vector, null, expected, 0, rows);

        float[] q4 = new float[rows];
        float[] q5 = new float[rows];
        GgufTensorOps.matVecRows(
                new GgufTensorOps.Q4KMatrix(columns, rows, blocksPerRow, quants, scales, emptyMins, false),
                vector,
                q4,
                rows,
                true);
        GgufTensorOps.matVecRows(
                new GgufTensorOps.Q5KMatrix(columns, rows, blocksPerRow, quants, scales, emptyMins, false),
                vector,
                q5,
                rows,
                true);

        assertArrayEquals(expected, q4, 0.0f);
        assertArrayEquals(expected, q5, 0.0f);
    }

    private static void assertTinyQ32NoBiasRows() {
        int rows = 4;
        int blocksPerRow = 2;
        int columns = blocksPerRow * GgufQuantFormats.Q4_0_BLOCK_SIZE;
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * blocksPerRow];
        fillPreparedRows(quants, scales, rows, columns, blocksPerRow, GgufQuantFormats.Q4_0_BLOCK_SIZE);
        float[] vector = patternedVector(columns, 17, 0.046875f);
        float[] expected = new float[rows];
        GgufTensorOps.Q32Matrix matrix =
                new GgufTensorOps.Q32Matrix(columns, rows, blocksPerRow, quants, scales, new float[0], false);

        GgufQ32Rows.fillMatVecRowsQ32(matrix, vector, null, expected, 0, rows);

        float[] actual = new float[rows];
        GgufTensorOps.matVecRows(matrix, vector, actual, rows, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyQ8Rows() {
        int rows = 4;
        int blocksPerRow = 2;
        int columns = blocksPerRow * GgufQuantFormats.Q8_0_BLOCK_SIZE;
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * blocksPerRow];
        fillPreparedRows(quants, scales, rows, columns, blocksPerRow, GgufQuantFormats.Q8_0_BLOCK_SIZE);
        float[] vector = patternedVector(columns, 19, 0.0390625f);
        float[] expected = new float[rows];
        GgufTensorOps.Q8Matrix matrix =
                new GgufTensorOps.Q8Matrix(columns, rows, blocksPerRow, GgufQuantFormats.Q8_0_BLOCK_SIZE, quants, scales);

        GgufQ8Rows.fillMatVecRowsQ8(matrix, vector, expected, 0, rows);

        float[] actual = new float[rows];
        GgufTensorOps.matVecRows(matrix, vector, actual, rows, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void fillPreparedRows(
            byte[] quants,
            float[] scales,
            int rows,
            int rowQuantStride,
            int rowGroupStride,
            int groupSize) {
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < rowGroupStride; group++) {
                int groupBase = row * rowGroupStride + group;
                int seed = 17 + row * 37 + group * 13;
                scales[groupBase] = ((group & 1) == 0 ? 0.625f : -0.75f) + row * 0.03125f;
                int quantBase = row * rowQuantStride + group * groupSize;
                for (int index = 0; index < groupSize; index++) {
                    quants[quantBase + index] = kPattern(seed, index);
                }
            }
        }
    }

    private static void fillPatternedQuants(byte[] quants, int blockSize, int[] seeds) {
        for (int block = 0; block < seeds.length; block++) {
            for (int index = 0; index < blockSize; index++) {
                quants[block * blockSize + index] = kPattern(seeds[block], index);
            }
        }
    }

    private static float[] patternedVector(int length, int period, float step) {
        float[] vector = new float[length];
        int midpoint = period / 2;
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % period - midpoint) * step;
        }
        return vector;
    }

    private static float referenceNoMin(int blockSize, float[] scales, int[] seeds, float[] vector) {
        float sum = 0.0f;
        int vectorBase = 0;
        for (int block = 0; block < scales.length; block++) {
            for (int index = 0; index < blockSize; index++) {
                sum += scales[block] * kPattern(seeds[block], index) * vector[vectorBase + index];
            }
            vectorBase += blockSize;
        }
        return sum;
    }

    private static byte kPattern(int seed, int index) {
        return (byte) (seed + index * 37);
    }
}
