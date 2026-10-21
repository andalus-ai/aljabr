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
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q2_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q3_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q6_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufRawKTest {
    @Test
    void rawSingleRowDispatchClassifiesKQuantMinStateAtRowOffset() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q2 = arena.allocate(2L * 84);
            MemorySegment q4 = arena.allocate(2L * 144);
            MemorySegment q5 = arena.allocate(2L * 176);
            writeQ2KBlock(q2.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeQ2KBlockWithMin(q2.asSlice(84, 84), (byte) 0x12, (byte) 0x55);
            writeQ4KNoMinLaneOrderBlock(q4.asSlice(0, 144));
            writeQ4KMinLaneOrderBlock(q4.asSlice(144, 144));
            writeQ5KNoMinLaneOrderBlock(q5.asSlice(0, 176));
            writeQ5KMinLaneOrderBlock(q5.asSlice(176, 176));

            float[] vector = ramp(256);
            float[] q2Sums = GgufSum.vector16GroupSums(vector, 256, new GgufTensorOps.Q4KWorkBuffer());
            float[] q4Sums = GgufSum.q4KVectorGroupSums(vector, 256, new GgufTensorOps.Q4KWorkBuffer());

            assertEquals(GgufQ2RawDot.dotRowQ2KNoMins(q2, 0, 256, vector, 0),
                    GgufRawDot.q2K(q2, 0, 256, vector, null), 0.0f);
            assertEquals(GgufQ2RawDot.dotRowQ2K(q2, 84, 256, vector, q2Sums),
                    GgufRawDot.q2K(q2, 84, 256, vector, null), 0.0f);
            assertEquals(GgufQ4RawDot.dotRowQ4KNoMins(q4, 0, 256, vector, 0),
                    GgufRawDot.q4K(q4, 0, 256, vector, null), 0.0f);
            assertEquals(GgufQ4RawDot.dotRowQ4KWithGroupSums(q4, 144, 256, vector, q4Sums),
                    GgufRawDot.q4K(q4, 144, 256, vector, null), 0.0f);
            assertEquals(GgufQ5RawDot.dotRowQ5KNoMins(q5, 0, 256, vector, 0),
                    GgufRawDot.q5K(q5, 0, 256, vector, null), 0.0f);
            assertEquals(GgufQ5RawDot.dotRowQ5KWithGroupSums(q5, 176, 256, vector, q4Sums),
                    GgufRawDot.q5K(q5, 176, 256, vector, null), 0.0f);
        }
    }

    @Test
    void rawKQuantRowOpsTreatZeroMinCodesAsNoMinsBeforeApplyingDMin() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84 + 144 + 176);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeSimpleQ4KBlock(segment.asSlice(84, 144));
            writeQ5KBlock(segment.asSlice(228, 176), (byte) 0xFF, (byte) 0);
            GGUFTensorInfo q2 = new GGUFTensorInfo("q2.raw.no_mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4 = new GGUFTensorInfo("q4.raw.no_mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5 = new GGUFTensorInfo("q5.raw.no_mins", new long[]{256, 1}, 13, 228, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q2, q4, q5), 0, segment, null);
            float[] ones = ones(256);
            float[] q2Expected = dequantizedRow(model, q2);
            float[] q4Expected = dequantizedRow(model, q4);
            float[] q5Expected = dequantizedRow(model, q5);
            float q2DotExpected = GgufTensorOps.dotRow(model, q2, 0, ones);
            float q4DotExpected = GgufTensorOps.dotRow(model, q4, 0, ones);
            float q5DotExpected = GgufTensorOps.dotRow(model, q5, 0, ones);

            segment.set(LE_SHORT, 82, (short) 0x7e00);
            segment.set(LE_SHORT, 86, (short) 0x7e00);
            segment.set(LE_SHORT, 230, (short) 0x7e00);

            assertArrayEquals(q2Expected, dequantizedRow(model, q2), 0.0f);
            assertArrayEquals(q4Expected, dequantizedRow(model, q4), 0.0f);
            assertArrayEquals(q5Expected, dequantizedRow(model, q5), 0.0f);
            assertEquals(q2DotExpected, GgufTensorOps.dotRow(model, q2, 0, ones), 0.0f);
            assertEquals(q4DotExpected, GgufTensorOps.dotRow(model, q4, 0, ones), 0.0f);
            assertEquals(q5DotExpected, GgufTensorOps.dotRow(model, q5, 0, ones), 0.0f);
        }
    }

    @Test
    void rawKNoMinFastPathsHandleUnrolledBlocksAndTail() {
        int blocks = 7;
        int columns = blocks * QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q2 = arena.allocate(blocks * (long) Q2_K_BLOCK_BYTES);
            MemorySegment q4 = arena.allocate(blocks * (long) Q4_K_BLOCK_BYTES);
            MemorySegment q5 = arena.allocate(blocks * (long) Q5_K_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeQ2KBlock(
                        q2.asSlice(block * (long) Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES),
                        (byte) (1 + (block & 3)),
                        (byte) (0x11 + block));
                writeSimpleQ4KBlock(q4.asSlice(block * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
                writeQ5KBlock(
                        q5.asSlice(block * (long) Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES),
                        (byte) (block * 0x11),
                        (byte) (0x10 + block));
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 31 - 15) * 0.03125f;
            }

            assertEquals(
                    GgufQ2RawDot.dotRowQ2K(q2, 0, columns, vector, 0),
                    GgufQ2RawDot.dotRowQ2KNoMins(q2, 0, columns, vector, 0),
                    1.0e-4f);
            assertEquals(
                    GgufQ4RawDot.dotRowQ4K(q4, 0, columns, vector, 0),
                    GgufQ4RawDot.dotRowQ4KNoMins(q4, 0, columns, vector, 0),
                    1.0e-4f);
            assertEquals(
                    GgufQ5RawDot.dotRowQ5K(q5, 0, columns, vector, 0),
                    GgufQ5RawDot.dotRowQ5KNoMins(q5, 0, columns, vector, 0),
                    1.0e-4f);
        }
    }

    @Test
    void rawKGroupSumFastPathsHandleUnrolledBlocksAndTail() {
        int blocks = 7;
        int columns = blocks * QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q2 = arena.allocate(blocks * (long) Q2_K_BLOCK_BYTES);
            MemorySegment q4 = arena.allocate(blocks * (long) Q4_K_BLOCK_BYTES);
            MemorySegment q5 = arena.allocate(blocks * (long) Q5_K_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeQ2KBlockWithMin(
                        q2.asSlice(block * (long) Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES),
                        (byte) ((((block % 3) + 1) << 4) | ((block % 5) + 1)),
                        (byte) (0x21 + block));
                writeQ4KBlockWithAllScalesAndMins(
                        q4.asSlice(block * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
                writeQ5KBlockWithMin(
                        q5.asSlice(block * (long) Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES),
                        (byte) (1 + block),
                        (byte) ((block % 3) + 1),
                        (byte) (0x11 * block),
                        (byte) (0x21 + block));
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 37 - 18) * 0.02734375f;
            }
            float[] vectorGroupSums =
                    GgufSum.q4KVectorGroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
            float[] vector16GroupSums =
                    GgufSum.vector16GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());

            assertEquals(
                    GgufQ2RawDot.dotRowQ2K(q2, 0, columns, vector, 0),
                    GgufQ2RawDot.dotRowQ2KWithGroupSums(q2, 0, columns, vector, vector16GroupSums),
                    1.0e-3f);
            assertEquals(
                    GgufQ2RawDot.dotRowQ2K(q2, 0, columns, vector, vector16GroupSums),
                    GgufQ2RawDot.dotRowQ2KWithGroupSums(q2, 0, columns, vector, vector16GroupSums),
                    0.0f);
            assertEquals(
                    GgufQ4RawDot.dotRowQ4K(q4, 0, columns, vector, 0),
                    GgufQ4RawDot.dotRowQ4KWithGroupSums(q4, 0, columns, vector, vectorGroupSums),
                    1.0e-3f);
            assertEquals(
                    GgufQ5RawDot.dotRowQ5K(q5, 0, columns, vector, 0),
                    GgufQ5RawDot.dotRowQ5KWithGroupSums(q5, 0, columns, vector, vectorGroupSums),
                    1.0e-3f);
        }
    }

    @Test
    void rawKTinyAffineMatVecSlicesUseDirectDots() {
        int rows = 4;
        int columns = QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q2 = arena.allocate(rows * (long) Q2_K_BLOCK_BYTES);
            MemorySegment q4 = arena.allocate(rows * (long) Q4_K_BLOCK_BYTES);
            MemorySegment q5 = arena.allocate(rows * (long) Q5_K_BLOCK_BYTES);
            MemorySegment q6 = arena.allocate(rows * (long) Q6_K_BLOCK_BYTES);
            for (int row = 0; row < rows; row++) {
                writeQ2KBlockWithMin(
                        q2.asSlice(row * (long) Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES),
                        (byte) (((row + 1) << 4) | (row + 2)),
                        (byte) (0x21 + row));
                writeQ4KBlockWithAllScalesAndMins(
                        q4.asSlice(row * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
                writeQ5KBlockWithMin(
                        q5.asSlice(row * (long) Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES),
                        (byte) (row + 1),
                        (byte) (row + 2),
                        (byte) (0x11 * row),
                        (byte) (0x31 + row));
                writeQ6KBlock(
                        q6.asSlice(row * (long) Q6_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES),
                        (byte) (0x41 + row),
                        (byte) (0x55 ^ row),
                        (byte) (row - 1));
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 29 - 14) * 0.0234375f;
            }

            assertTinyDirectQ2K(q2, rows, columns, vector);
            assertTinyDirectQ4K(q4, rows, columns, vector);
            assertTinyDirectQ5K(q5, rows, columns, vector);
            assertTinyDirectQ6K(q6, rows, columns, vector);
        }
    }

    @Test
    void rawKTinyAffineMatVecSlicesBypassWorkerProbe() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        GgufRows.clearRawWorkerAccessCache();
        int rows = 4;
        int columns = QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q6 = arena.allocate(rows * (long) Q6_K_BLOCK_BYTES);
            for (int row = 0; row < rows; row++) {
                writeQ6KBlock(
                        q6.asSlice(row * (long) Q6_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES),
                        (byte) (0x31 + row),
                        (byte) (0x55 ^ row),
                        (byte) (row - 2));
            }

            float[] vector = ones(columns);
            float[] expected = new float[rows];
            float[] actual = new float[rows];
            for (int row = 0; row < rows; row++) {
                expected[row] = GgufQ6RawDot.dotRowQ6K(
                        q6, row * (long) Q6_K_BLOCK_BYTES, columns, vector, 0);
            }
            GgufRawMatVec.q6K(q6, columns, Q6_K_BLOCK_BYTES, vector, actual, rows, true);

            assertArrayEquals(expected, actual, 0.0f);
            assertEquals(0, GgufRows.rawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessFastCacheSize());
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
            GgufRows.clearRawWorkerAccessCache();
        }
    }

    @Test
    void rawKTinyNoMinMatVecSlicesBypassWorkerProbe() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        GgufRows.clearRawWorkerAccessCache();
        int rows = 4;
        int columns = QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q2 = arena.allocate(rows * (long) Q2_K_BLOCK_BYTES);
            MemorySegment q3 = arena.allocate(rows * (long) Q3_K_BLOCK_BYTES);
            MemorySegment q4 = arena.allocate(rows * (long) Q4_K_BLOCK_BYTES);
            MemorySegment q5 = arena.allocate(rows * (long) Q5_K_BLOCK_BYTES);
            for (int row = 0; row < rows; row++) {
                writeQ2KNoMinLaneOrderBlock(q2.asSlice(row * (long) Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES));
                writeQ3KLaneOrderBlock(q3.asSlice(row * (long) Q3_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES));
                writeQ4KNoMinLaneOrderBlock(q4.asSlice(row * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
                writeQ5KNoMinLaneOrderBlock(q5.asSlice(row * (long) Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES));
            }

            float[] vector = ones(columns);
            assertTinyNoMinQ2K(q2, rows, columns, vector);
            assertTinyNoMinQ3K(q3, rows, columns, vector);
            assertTinyNoMinQ4K(q4, rows, columns, vector);
            assertTinyNoMinQ5K(q5, rows, columns, vector);

            assertEquals(0, GgufRows.rawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessCacheSize());
            assertEquals(0, GgufRows.recentRawWorkerAccessFastCacheSize());
        } finally {
            restoreProperty("aljabr.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("aljabr.gguf.parallel_threads", previousThreads);
            restoreProperty("aljabr.gguf.parallel_chunks_per_thread", previousChunks);
            GgufParallelConfig.resetParallelConfig();
            GgufRows.clearRawWorkerAccessCache();
        }
    }

    @Test
    void rawKRowWalkersHandleUnrolledRowsAndTail() {
        int rows = 7;
        int columns = QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q2NoMin = arena.allocate(rows * (long) Q2_K_BLOCK_BYTES);
            MemorySegment q2Mins = arena.allocate(rows * (long) Q2_K_BLOCK_BYTES);
            MemorySegment q3 = arena.allocate(rows * (long) Q3_K_BLOCK_BYTES);
            MemorySegment q4NoMin = arena.allocate(rows * (long) Q4_K_BLOCK_BYTES);
            MemorySegment q4Mins = arena.allocate(rows * (long) Q4_K_BLOCK_BYTES);
            MemorySegment q5NoMin = arena.allocate(rows * (long) Q5_K_BLOCK_BYTES);
            MemorySegment q5Mins = arena.allocate(rows * (long) Q5_K_BLOCK_BYTES);
            MemorySegment q6 = arena.allocate(rows * (long) Q6_K_BLOCK_BYTES);
            for (int row = 0; row < rows; row++) {
                writeQ2KBlock(
                        q2NoMin.asSlice(row * (long) Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES),
                        (byte) (1 + row),
                        (byte) (0x11 + row));
                writeQ2KBlockWithMin(
                        q2Mins.asSlice(row * (long) Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES),
                        (byte) ((((row % 3) + 1) << 4) | ((row % 5) + 1)),
                        (byte) (0x21 + row));
                writeQ3KBlock(
                        q3.asSlice(row * (long) Q3_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES),
                        row - 3,
                        (byte) (0x31 + row),
                        (byte) (0x11 * (row + 1)));
                writeSimpleQ4KBlock(q4NoMin.asSlice(row * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
                writeQ4KBlockWithAllScalesAndMins(
                        q4Mins.asSlice(row * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
                writeQ5KBlock(
                        q5NoMin.asSlice(row * (long) Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES),
                        (byte) (0x11 * row),
                        (byte) (0x10 + row));
                writeQ5KBlockWithMin(
                        q5Mins.asSlice(row * (long) Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES),
                        (byte) (1 + row),
                        (byte) ((row % 3) + 1),
                        (byte) (0x11 * row),
                        (byte) (0x21 + row));
                writeQ6KBlock(
                        q6.asSlice(row * (long) Q6_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES),
                        (byte) (0x31 + row),
                        (byte) (0x55 ^ row),
                        (byte) (row - 2));
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 41 - 20) * 0.01953125f;
            }
            float[] vector16GroupSums =
                    GgufSum.vector16GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
            float[] vector32GroupSums =
                    GgufSum.q4KVectorGroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());

            assertRawKRowsQ2NoMins(q2NoMin, rows, columns, vector);
            assertRawKRowsQ2Mins(q2Mins, rows, columns, vector, vector16GroupSums);
            assertRawKRowsQ3(q3, rows, columns, vector);
            assertRawKRowsQ4NoMins(q4NoMin, rows, columns, vector);
            assertRawKRowsQ4Mins(q4Mins, rows, columns, vector, vector32GroupSums);
            assertRawKRowsQ5NoMins(q5NoMin, rows, columns, vector);
            assertRawKRowsQ5Mins(q5Mins, rows, columns, vector, vector32GroupSums);
            assertRawKRowsQ6(q6, rows, columns, vector, vector16GroupSums);
        }
    }

    private static void assertTinyDirectQ2K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufRawDot.q2K(
                    segment, row * (long) Q2_K_BLOCK_BYTES, columns, vector, true);
        }
        GgufRawMatVec.q2K(segment, columns, Q2_K_BLOCK_BYTES, vector, actual, rows, true, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyDirectQ4K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufRawDot.q4K(
                    segment, row * (long) Q4_K_BLOCK_BYTES, columns, vector, true);
        }
        GgufRawMatVec.q4K(segment, columns, Q4_K_BLOCK_BYTES, vector, actual, rows, true, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyDirectQ5K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufRawDot.q5K(
                    segment, row * (long) Q5_K_BLOCK_BYTES, columns, vector, true);
        }
        GgufRawMatVec.q5K(segment, columns, Q5_K_BLOCK_BYTES, vector, actual, rows, true, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyDirectQ6K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ6RawDot.dotRowQ6K(
                    segment, row * (long) Q6_K_BLOCK_BYTES, columns, vector, 0);
        }
        GgufRawMatVec.q6K(segment, columns, Q6_K_BLOCK_BYTES, vector, actual, rows, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyNoMinQ2K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufKRawRows.fillMatVecRowsQ2KNoMins(segment, columns, Q2_K_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q2K(segment, columns, Q2_K_BLOCK_BYTES, vector, actual, rows, true, false);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyNoMinQ3K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufKRawRows.fillMatVecRowsQ3K(segment, columns, Q3_K_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q3K(segment, columns, Q3_K_BLOCK_BYTES, vector, actual, rows, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyNoMinQ4K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufKRawRows.fillMatVecRowsQ4KNoMins(segment, columns, Q4_K_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q4K(segment, columns, Q4_K_BLOCK_BYTES, vector, actual, rows, true, false);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyNoMinQ5K(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufKRawRows.fillMatVecRowsQ5KNoMins(segment, columns, Q5_K_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q5K(segment, columns, Q5_K_BLOCK_BYTES, vector, actual, rows, true, false);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ2NoMins(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ2RawDot.dotRowQ2KNoMins(
                    segment, row * (long) Q2_K_BLOCK_BYTES, columns, vector, 0);
        }
        GgufKRawRows.fillMatVecRowsQ2KNoMins(segment, columns, Q2_K_BLOCK_BYTES, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ2Mins(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ2RawDot.dotRowQ2KWithGroupSums(
                    segment, row * (long) Q2_K_BLOCK_BYTES, columns, vector, vectorGroupSums);
        }
        GgufKRawRows.fillMatVecRowsQ2K(
                segment, columns, Q2_K_BLOCK_BYTES, vector, vectorGroupSums, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ3(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        int[] scales = new int[16];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ3RawDot.dotRowQ3K(
                    segment, row * (long) Q3_K_BLOCK_BYTES, columns, vector, 0, scales);
        }
        GgufKRawRows.fillMatVecRowsQ3K(segment, columns, Q3_K_BLOCK_BYTES, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ4NoMins(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ4RawDot.dotRowQ4KNoMins(
                    segment, row * (long) Q4_K_BLOCK_BYTES, columns, vector, 0);
        }
        GgufKRawRows.fillMatVecRowsQ4KNoMins(segment, columns, Q4_K_BLOCK_BYTES, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ4Mins(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ4RawDot.dotRowQ4KWithGroupSums(
                    segment, row * (long) Q4_K_BLOCK_BYTES, columns, vector, vectorGroupSums);
        }
        GgufKRawRows.fillMatVecRowsQ4K(
                segment, columns, Q4_K_BLOCK_BYTES, vector, vectorGroupSums, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ5NoMins(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ5RawDot.dotRowQ5KNoMins(
                    segment, row * (long) Q5_K_BLOCK_BYTES, columns, vector, 0);
        }
        GgufKRawRows.fillMatVecRowsQ5KNoMins(segment, columns, Q5_K_BLOCK_BYTES, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ5Mins(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ5RawDot.dotRowQ5KWithGroupSums(
                    segment, row * (long) Q5_K_BLOCK_BYTES, columns, vector, vectorGroupSums);
        }
        GgufKRawRows.fillMatVecRowsQ5K(
                segment, columns, Q5_K_BLOCK_BYTES, vector, vectorGroupSums, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawKRowsQ6(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ6RawDot.dotRowQ6KWithGroupSums(
                    segment, row * (long) Q6_K_BLOCK_BYTES, columns, vector, vectorGroupSums);
        }
        GgufKRawRows.fillMatVecRowsQ6K(
                segment, columns, Q6_K_BLOCK_BYTES, vector, vectorGroupSums, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }
}
