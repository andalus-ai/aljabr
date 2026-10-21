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
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_GROUPS_PER_BLOCK;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;

class GgufKPrepMinTest {
    @Test
    void preparedKSingleRowMinMatVecMatchesDotRow() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84 + 144 + 176);
            writeQ2KMinLaneOrderBlock(segment.asSlice(0, 84));
            writeQ4KMinLaneOrderBlock(segment.asSlice(84, 144));
            writeQ5KMinLaneOrderBlock(segment.asSlice(228, 176));
            GGUFTensorInfo q2Tensor = new GGUFTensorInfo("q2.prepared.single.mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4.prepared.single.mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5.prepared.single.mins", new long[]{256, 1}, 13, 228, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q2Tensor, q4Tensor, q5Tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.Q2KMatrix q2Matrix = GgufTensorOps.q2KMatrix(model, q2Tensor);
            GgufTensorOps.Q4KMatrix q4Matrix = GgufTensorOps.q4KMatrix(model, q4Tensor);
            GgufTensorOps.Q5KMatrix q5Matrix = GgufTensorOps.q5KMatrix(model, q5Tensor);

            assertTrue(q2Matrix.hasGroupMins());
            assertTrue(q4Matrix.hasGroupMins());
            assertTrue(q5Matrix.hasGroupMins());

            GgufTensorOps.matVecRows(q2Matrix, vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q2Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(q4Matrix, vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q4Tensor, 0, vector), output[0], 0.0f);
            GgufTensorOps.matVecRows(q5Matrix, vector, output, 1, true);
            assertEquals(GgufTensorOps.dotRow(model, q5Tensor, 0, vector), output[0], 0.0f);
        }
    }

    @Test
    void preparedKDirectMinDotsMatchPrecomputedVectorSums() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84 + 144 + 176);
            writeQ2KMinLaneOrderBlock(segment.asSlice(0, 84));
            writeQ4KMinLaneOrderBlock(segment.asSlice(84, 144));
            writeQ5KMinLaneOrderBlock(segment.asSlice(228, 176));
            GGUFTensorInfo q2Tensor = new GGUFTensorInfo("q2.prepared.direct.mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4.prepared.direct.mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5.prepared.direct.mins", new long[]{256, 1}, 13, 228, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q2Tensor, q4Tensor, q5Tensor), 0, segment, null);
            float[] vector = fractionalRamp(256);
            float[] q2VectorSums =
                    GgufSum.vector16GroupSums(vector, 256, new GgufTensorOps.Q4KWorkBuffer());
            float[] k32VectorSums =
                    GgufSum.q4KVectorGroupSums(vector, 256, new GgufTensorOps.Q4KWorkBuffer());
            GgufTensorOps.Q2KMatrix q2Matrix = GgufTensorOps.q2KMatrix(model, q2Tensor);
            GgufTensorOps.Q4KMatrix q4Matrix = GgufTensorOps.q4KMatrix(model, q4Tensor);
            GgufTensorOps.Q5KMatrix q5Matrix = GgufTensorOps.q5KMatrix(model, q5Tensor);

            assertEquals(
                    GgufKRows.dotRowQ2K(
                            q2Matrix.quantStride(),
                            q2Matrix.groupStride(),
                            q2Matrix.quants(),
                            q2Matrix.groupScales(),
                            q2Matrix.groupMins(),
                            vector,
                            q2VectorSums,
                            0),
                    GgufKRows.dotRowQ2KDirect(
                            q2Matrix.quantStride(),
                            q2Matrix.groupStride(),
                            q2Matrix.quants(),
                            q2Matrix.groupScales(),
                            q2Matrix.groupMins(),
                            vector,
                            0),
                    1.0e-4f);
            assertEquals(
                    GgufKRows.dotRowK32Prepared(
                            q4Matrix.quantStride(),
                            q4Matrix.groupStride(),
                            q4Matrix.quants(),
                            q4Matrix.groupScales(),
                            q4Matrix.groupMins(),
                            vector,
                            k32VectorSums,
                            0),
                    GgufKRows.dotRowK32PreparedDirect(
                            q4Matrix.quantStride(),
                            q4Matrix.groupStride(),
                            q4Matrix.quants(),
                            q4Matrix.groupScales(),
                            q4Matrix.groupMins(),
                            vector,
                            0),
                    1.0e-4f);
            assertEquals(
                    GgufKRows.dotRowK32Prepared(
                            q5Matrix.quantStride(),
                            q5Matrix.groupStride(),
                            q5Matrix.quants(),
                            q5Matrix.groupScales(),
                            q5Matrix.groupMins(),
                            vector,
                            k32VectorSums,
                            0),
                    GgufKRows.dotRowK32PreparedDirect(
                            q5Matrix.quantStride(),
                            q5Matrix.groupStride(),
                            q5Matrix.quants(),
                            q5Matrix.groupScales(),
                            q5Matrix.groupMins(),
                            vector,
                            0),
                    1.0e-4f);
        }
    }

    @Test
    void preparedKDirectMinVectorAccumulatorsMatchIndependentReferences() {
        int groups16 = 7;
        byte[] quants16 = new byte[groups16 * 16];
        float[] scales16 = {1.0f, -0.5f, 2.0f, -1.25f, 0.75f, -0.875f, 1.625f};
        float[] mins16 = {0.5f, -0.25f, 1.5f, -0.75f, 0.375f, 1.125f, -0.625f};
        int[] seeds16 = {17, 47, 83, 127, 173, 211, 251};
        fillPatternedQuants(quants16, 16, seeds16);
        float[] vector16 = patternedVector(groups16 * 16, 19, 0.0625f);
        float[] vectorSums16 = groupSums(16, groups16, vector16);

        float expected16 = referenceMin(16, scales16, mins16, seeds16, vector16);
        assertEquals(
                expected16,
                GgufKDot.dotRowQ2KPreparedDirectScalar(groups16, quants16, scales16, mins16, 0, 0, vector16),
                1.0e-3f);
        assertEquals(
                expected16,
                GgufKDot.dotRowQ2KPreparedScalar(
                        groups16, quants16, scales16, mins16, 0, 0, vector16, vectorSums16),
                1.0e-3f);
        assertEquals(
                expected16,
                GgufKDot.dotRowQ2KPreparedVector(
                        groups16, quants16, scales16, mins16, 0, 0, vector16, vectorSums16),
                1.0e-3f);
        assertEquals(
                expected16,
                GgufKDot.dotRowQ2KPreparedDirectVector(groups16, quants16, scales16, mins16, 0, 0, vector16),
                1.0e-3f);

        int groups32 = 7;
        byte[] quants32 = new byte[groups32 * 32];
        float[] scales32 = {0.75f, -1.5f, 2.25f, -0.625f, 1.375f, -1.125f, 0.3125f};
        float[] mins32 = {0.25f, -0.875f, 1.125f, 0.5f, -0.375f, 0.9375f, -1.25f};
        int[] seeds32 = {29, 71, 131, 181, 223, 17, 97};
        fillPatternedQuants(quants32, 32, seeds32);
        float[] vector32 = patternedVector(groups32 * 32, 23, 0.09375f);
        float[] vectorSums32 = groupSums(32, groups32, vector32);

        float expected32 = referenceMin(32, scales32, mins32, seeds32, vector32);
        assertEquals(
                expected32,
                GgufKDot.dotRowK32GroupsPreparedDirectScalar(groups32, quants32, scales32, mins32, 0, 0, vector32),
                1.0e-3f);
        assertEquals(
                expected32,
                GgufKDot.dotRowK32GroupsPreparedScalar(
                        groups32, quants32, scales32, mins32, 0, 0, vector32, vectorSums32),
                1.0e-3f);
        assertEquals(
                expected32,
                GgufKDot.dotRowK32GroupsPreparedVector(
                        groups32, quants32, scales32, mins32, 0, 0, vector32, vectorSums32),
                1.0e-3f);
        assertEquals(
                expected32,
                GgufKDot.dotRowK32GroupsPreparedDirectVector(groups32, quants32, scales32, mins32, 0, 0, vector32),
                1.0e-3f);
    }

    @Test
    void preparedQ2KMinRowWalkersHandleUnrolledRowsAndTail() {
        int rows = 7;
        int blocksPerRow = 2;
        int columns = blocksPerRow * QK_K;
        int groupsPerRow = blocksPerRow * QK_GROUPS_PER_BLOCK;
        byte[] quants = new byte[rows * columns];
        float[] scales = new float[rows * groupsPerRow];
        float[] mins = new float[rows * groupsPerRow];
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groupsPerRow; group++) {
                int groupBase = row * groupsPerRow + group;
                int seed = 11 + row * 43 + group * 13;
                scales[groupBase] = ((group & 1) == 0 ? 0.5f : -0.71875f) + row * 0.01953125f;
                mins[groupBase] = ((row & 1) == 0 ? 0.3125f : -0.4375f) + group * 0.01171875f;
                int quantBase = row * columns + group * 16;
                for (int index = 0; index < 16; index++) {
                    quants[quantBase + index] = kPattern(seed, index);
                }
            }
        }
        float[] vector = patternedVector(columns, 41, 0.0234375f);
        float[] vectorGroupSums =
                GgufSum.vector16GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
        float[] expectedPrecomputed = new float[rows];
        float[] actualPrecomputed = new float[rows];
        float[] expectedDirect = new float[rows];
        float[] actualDirect = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedPrecomputed[row] =
                    GgufKRows.dotRowQ2K(blocksPerRow, quants, scales, mins, vector, vectorGroupSums, row);
            expectedDirect[row] =
                    GgufKRows.dotRowQ2KDirect(blocksPerRow, quants, scales, mins, vector, row);
        }

        GgufKRows.fillMatVecRowsQ2K(
                blocksPerRow, quants, scales, mins, vector, vectorGroupSums, actualPrecomputed, 0, rows);
        GgufKRows.fillMatVecRowsQ2KDirect(blocksPerRow, quants, scales, mins, vector, actualDirect, 0, rows);

        assertArrayEquals(expectedPrecomputed, actualPrecomputed, 0.0f);
        assertArrayEquals(expectedDirect, actualDirect, 0.0f);

        float[] expectedPrecomputedScalar = new float[rows];
        float[] actualPrecomputedScalar = new float[rows];
        float[] expectedDirectScalar = new float[rows];
        float[] actualDirectScalar = new float[rows];
        float[] actualPrecomputedKernelVector = new float[rows];
        float[] actualPrecomputedKernelScalar = new float[rows];
        float[] actualDirectKernelVector = new float[rows];
        float[] actualDirectKernelScalar = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedPrecomputedScalar[row] = GgufKDot.dotRowQ2KPreparedScalar(
                    groupsPerRow, quants, scales, mins, row * columns, row * groupsPerRow, vector, vectorGroupSums);
            expectedDirectScalar[row] = GgufKDot.dotRowQ2KPreparedDirectScalar(
                    groupsPerRow, quants, scales, mins, row * columns, row * groupsPerRow, vector);
        }
        fillQ2KScalar4(groupsPerRow, quants, scales, mins, columns, vector, vectorGroupSums,
                actualPrecomputedScalar, rows);
        fillQ2KDirectScalar4(groupsPerRow, quants, scales, mins, columns, vector, actualDirectScalar, rows);
        GgufKRows.fillMatVecRowsQ2K(
                GgufTensorOps.Q2KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_VECTOR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                vectorGroupSums,
                actualPrecomputedKernelVector,
                0,
                rows);
        GgufKRows.fillMatVecRowsQ2K(
                GgufTensorOps.Q2KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_SCALAR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                vectorGroupSums,
                actualPrecomputedKernelScalar,
                0,
                rows);
        GgufKRows.fillMatVecRowsQ2KDirect(
                GgufTensorOps.Q2KMatrix.ROW_KERNEL_DIRECT_MINS_VECTOR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                actualDirectKernelVector,
                0,
                rows);
        GgufKRows.fillMatVecRowsQ2KDirect(
                GgufTensorOps.Q2KMatrix.ROW_KERNEL_DIRECT_MINS_SCALAR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                actualDirectKernelScalar,
                0,
                rows);

        assertArrayEquals(expectedPrecomputedScalar, actualPrecomputedScalar, 0.0f);
        assertArrayEquals(expectedDirectScalar, actualDirectScalar, 0.0f);
        assertArrayEquals(expectedPrecomputedScalar, actualPrecomputedKernelVector, 1.0e-3f);
        assertArrayEquals(expectedPrecomputedScalar, actualPrecomputedKernelScalar, 0.0f);
        assertArrayEquals(expectedDirectScalar, actualDirectKernelVector, 1.0e-3f);
        assertArrayEquals(expectedDirectScalar, actualDirectKernelScalar, 0.0f);
        assertEquals(
                expectedPrecomputedScalar[rows - 1],
                GgufKRows.dotRowQ2K(
                        GgufTensorOps.Q2KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_VECTOR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        mins,
                        vector,
                        vectorGroupSums,
                        rows - 1),
                1.0e-3f);
        assertEquals(
                expectedDirectScalar[rows - 1],
                GgufKRows.dotRowQ2KDirect(
                        GgufTensorOps.Q2KMatrix.ROW_KERNEL_DIRECT_MINS_VECTOR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        mins,
                        vector,
                        rows - 1),
                1.0e-3f);
    }

    @Test
    void preparedK32MinScalarFourRowReducersMatchSingleRowReferences() {
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
                int seed = 23 + row * 31 + group * 17;
                scales[groupBase] = ((group & 1) == 0 ? 0.6875f : -0.40625f) + row * 0.015625f;
                mins[groupBase] = ((row & 1) == 0 ? 0.21875f : -0.53125f) + group * 0.0078125f;
                int quantBase = row * columns + group * 32;
                for (int index = 0; index < 32; index++) {
                    quants[quantBase + index] = kPattern(seed, index);
                }
            }
        }
        float[] vector = patternedVector(columns, 31, 0.03125f);
        float[] vectorGroupSums =
                GgufSum.q4KVectorGroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
        float[] expectedPrecomputedScalar = new float[rows];
        float[] actualPrecomputedScalar = new float[rows];
        float[] expectedDirectScalar = new float[rows];
        float[] actualDirectScalar = new float[rows];
        float[] actualPrecomputedKernelVector = new float[rows];
        float[] actualPrecomputedKernelScalar = new float[rows];
        float[] actualDirectKernelVector = new float[rows];
        float[] actualDirectKernelScalar = new float[rows];
        for (int row = 0; row < rows; row++) {
            expectedPrecomputedScalar[row] = GgufKDot.dotRowK32GroupsPreparedScalar(
                    groupsPerRow, quants, scales, mins, row * columns, row * groupsPerRow, vector, vectorGroupSums);
            expectedDirectScalar[row] = GgufKDot.dotRowK32GroupsPreparedDirectScalar(
                    groupsPerRow, quants, scales, mins, row * columns, row * groupsPerRow, vector);
        }

        fillK32Scalar4(groupsPerRow, quants, scales, mins, columns, vector, vectorGroupSums,
                actualPrecomputedScalar, rows);
        fillK32DirectScalar4(groupsPerRow, quants, scales, mins, columns, vector, actualDirectScalar, rows);
        GgufKRows.fillMatVecRowsK32Prepared(
                GgufTensorOps.Q4KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_VECTOR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                vectorGroupSums,
                actualPrecomputedKernelVector,
                0,
                rows);
        GgufKRows.fillMatVecRowsK32Prepared(
                GgufTensorOps.Q4KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_SCALAR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                vectorGroupSums,
                actualPrecomputedKernelScalar,
                0,
                rows);
        GgufKRows.fillMatVecRowsK32PreparedDirect(
                GgufTensorOps.Q4KMatrix.ROW_KERNEL_DIRECT_MINS_VECTOR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                actualDirectKernelVector,
                0,
                rows);
        GgufKRows.fillMatVecRowsK32PreparedDirect(
                GgufTensorOps.Q4KMatrix.ROW_KERNEL_DIRECT_MINS_SCALAR,
                columns,
                groupsPerRow,
                quants,
                scales,
                mins,
                vector,
                actualDirectKernelScalar,
                0,
                rows);

        assertArrayEquals(expectedPrecomputedScalar, actualPrecomputedScalar, 0.0f);
        assertArrayEquals(expectedDirectScalar, actualDirectScalar, 0.0f);
        assertArrayEquals(expectedPrecomputedScalar, actualPrecomputedKernelVector, 1.0e-3f);
        assertArrayEquals(expectedPrecomputedScalar, actualPrecomputedKernelScalar, 0.0f);
        assertArrayEquals(expectedDirectScalar, actualDirectKernelVector, 1.0e-3f);
        assertArrayEquals(expectedDirectScalar, actualDirectKernelScalar, 0.0f);
        assertEquals(
                expectedPrecomputedScalar[rows - 1],
                GgufKRows.dotRowK32Prepared(
                        GgufTensorOps.Q5KMatrix.ROW_KERNEL_PRECOMPUTED_MINS_VECTOR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        mins,
                        vector,
                        vectorGroupSums,
                        rows - 1),
                1.0e-3f);
        assertEquals(
                expectedDirectScalar[rows - 1],
                GgufKRows.dotRowK32PreparedDirect(
                        GgufTensorOps.Q5KMatrix.ROW_KERNEL_DIRECT_MINS_VECTOR,
                        columns,
                        groupsPerRow,
                        quants,
                        scales,
                        mins,
                        vector,
                        rows - 1),
                1.0e-3f);
    }

    private static void fillQ2KScalar4(
            int groupsPerRow,
            byte[] quants,
            float[] scales,
            float[] mins,
            int rowQuantStride,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufKDot.dotRowsQ2KPreparedScalar4(
                    groupsPerRow,
                    quants,
                    scales,
                    mins,
                    row * rowQuantStride,
                    rowQuantStride,
                    row * groupsPerRow,
                    groupsPerRow,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufKDot.dotRowQ2KPreparedScalar(
                    groupsPerRow, quants, scales, mins, row * rowQuantStride, row * groupsPerRow, vector,
                    vectorGroupSums);
        }
    }

    private static void fillQ2KDirectScalar4(
            int groupsPerRow,
            byte[] quants,
            float[] scales,
            float[] mins,
            int rowQuantStride,
            float[] vector,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufKDot.dotRowsQ2KPreparedDirectScalar4(
                    groupsPerRow,
                    quants,
                    scales,
                    mins,
                    row * rowQuantStride,
                    rowQuantStride,
                    row * groupsPerRow,
                    groupsPerRow,
                    vector,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufKDot.dotRowQ2KPreparedDirectScalar(
                    groupsPerRow, quants, scales, mins, row * rowQuantStride, row * groupsPerRow, vector);
        }
    }

    private static void fillK32Scalar4(
            int groupsPerRow,
            byte[] quants,
            float[] scales,
            float[] mins,
            int rowQuantStride,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufKDot.dotRowsK32GroupsPreparedScalar4(
                    groupsPerRow,
                    quants,
                    scales,
                    mins,
                    row * rowQuantStride,
                    rowQuantStride,
                    row * groupsPerRow,
                    groupsPerRow,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufKDot.dotRowK32GroupsPreparedScalar(
                    groupsPerRow, quants, scales, mins, row * rowQuantStride, row * groupsPerRow, vector,
                    vectorGroupSums);
        }
    }

    private static void fillK32DirectScalar4(
            int groupsPerRow,
            byte[] quants,
            float[] scales,
            float[] mins,
            int rowQuantStride,
            float[] vector,
            float[] output,
            int rows) {
        int row = 0;
        int tailStart = rows - (rows & 3);
        for (; row < tailStart; row += 4) {
            GgufKDot.dotRowsK32GroupsPreparedDirectScalar4(
                    groupsPerRow,
                    quants,
                    scales,
                    mins,
                    row * rowQuantStride,
                    rowQuantStride,
                    row * groupsPerRow,
                    groupsPerRow,
                    vector,
                    output,
                    row);
        }
        for (; row < rows; row++) {
            output[row] = GgufKDot.dotRowK32GroupsPreparedDirectScalar(
                    groupsPerRow, quants, scales, mins, row * rowQuantStride, row * groupsPerRow, vector);
        }
    }

    private static float[] fractionalRamp(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i + 1) * 0.0625f + ((i & 1) == 0 ? 0.25f : -0.125f);
        }
        return values;
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

    private static float[] groupSums(int blockSize, int groups, float[] vector) {
        float[] sums = new float[groups];
        int vectorBase = 0;
        for (int group = 0; group < groups; group++) {
            float sum = 0.0f;
            for (int index = 0; index < blockSize; index++) {
                sum += vector[vectorBase + index];
            }
            sums[group] = sum;
            vectorBase += blockSize;
        }
        return sums;
    }

    private static float referenceMin(int blockSize, float[] scales, float[] mins, int[] seeds, float[] vector) {
        float sum = 0.0f;
        int vectorBase = 0;
        for (int block = 0; block < scales.length; block++) {
            for (int index = 0; index < blockSize; index++) {
                sum += (scales[block] * kPattern(seeds[block], index) - mins[block])
                        * vector[vectorBase + index];
            }
            vectorBase += blockSize;
        }
        return sum;
    }

    private static byte kPattern(int seed, int index) {
        return (byte) (seed + index * 37);
    }
}
