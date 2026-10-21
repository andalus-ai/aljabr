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
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class GgufQ32RawTest {
    @Test
    void rawQ32MatVecStreamsWhenPreparedCacheIsTooSmall() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "40");
        GgufRawPathHints.clearRecentHintCache();
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4_0", new long[]{32, 2}, 2, 0, 2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertFalse(GgufTensorOps.shouldUsePreparedMatrixCache(tensor, 2, 1, 40));
            GgufTensorOps.PreparedMatrixCachePlan plan = GgufTensorOps.planPreparedMatrixCaches(model);
            assertEquals(0, plan.preparedCandidates());
            assertEquals(1, plan.skippedCacheTooSmallTensors());
            assertEquals(0L, plan.estimatedPreparedBytes());

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(32), output, 2, true);
            assertEquals(16.0f, output[0], 0.0f);
            assertEquals(48.0f, output[1], 0.0f);
            assertEquals(0, GgufRawPathHints.recentHintCacheSize());
            assertEquals(0, GgufRawPathHints.recentHintFastCacheSize());
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));

            GgufTensorOps.PreparedMatrixCacheStats stats = GgufTensorOps.prepareMatrixCaches(model);
            assertEquals(0, stats.preparedCandidates());
            assertEquals(0, stats.preparedTensors());
            assertEquals(1, stats.skippedCacheTooSmallTensors());
            assertEquals(0, stats.cacheEntries());
            assertEquals(0L, stats.cacheBytes());
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previousMaxBytes);
            GgufRawPathHints.clearRecentHintCache();
        }
    }

    @Test
    void rawQ32NoBiasDotsHandleUnrolledBlocksAndTail() {
        int blocks = 5;
        short[] halfScales = {(short) 0x3c00, (short) 0x4000, (short) 0xbc00, (short) 0x3800, (short) 0x4200};
        float[] scales = {1.0f, 2.0f, -1.0f, 0.5f, 3.0f};
        byte[] packedQuants = {0x21, 0x43, 0x65, (byte) 0x87, (byte) 0xa9};
        int[] highBits = {0, 0x5555_5555, -1, 0x0f0f_f0f0, 0x3333_cccc};
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4 = arena.allocate(blocks * (long) Q4_0_BLOCK_BYTES);
            MemorySegment q5 = arena.allocate(blocks * (long) Q5_0_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeQ4_0Block(
                        q4.asSlice(block * (long) Q4_0_BLOCK_BYTES, Q4_0_BLOCK_BYTES),
                        halfScales[block],
                        packedQuants[block]);
                writeQ5_0Block(
                        q5.asSlice(block * (long) Q5_0_BLOCK_BYTES, Q5_0_BLOCK_BYTES),
                        halfScales[block],
                        highBits[block],
                        packedQuants[block]);
            }

            float[] vector = patternedVector(blocks * Q4_0_BLOCK_SIZE);

            assertEquals(
                    referenceQ4_0(scales, packedQuants, vector),
                    GgufQ32RawDot.dotRowQ4_0(q4, 0, vector.length, vector, 0),
                    1.0e-3f);
            assertEquals(
                    referenceQ5_0(scales, packedQuants, highBits, vector),
                    GgufQ32RawDot.dotRowQ5_0(q5, 0, vector.length, vector, 0),
                    1.0e-3f);
        }
    }

    @Test
    void rawQ32RowWalkersHandleUnrolledRowsAndTail() {
        int rows = 7;
        int blocks = 5;
        int columns = blocks * Q4_0_BLOCK_SIZE;
        long q4_0RowBytes = blocks * (long) Q4_0_BLOCK_BYTES;
        long q4_1RowBytes = blocks * (long) Q4_1_BLOCK_BYTES;
        long q5_0RowBytes = blocks * (long) Q5_0_BLOCK_BYTES;
        long q5_1RowBytes = blocks * (long) Q5_1_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4_0 = arena.allocate(rows * q4_0RowBytes);
            MemorySegment q4_1 = arena.allocate(rows * q4_1RowBytes);
            MemorySegment q4_1NoBias = arena.allocate(rows * q4_1RowBytes);
            MemorySegment q5_0 = arena.allocate(rows * q5_0RowBytes);
            MemorySegment q5_1 = arena.allocate(rows * q5_1RowBytes);
            MemorySegment q5_1NoBias = arena.allocate(rows * q5_1RowBytes);
            for (int row = 0; row < rows; row++) {
                for (int block = 0; block < blocks; block++) {
                    short scale = (short) (0x3800 + (((row + block) & 3) << 10));
                    short min = (short) (0x3000 + (((row + block) % 5) << 8));
                    byte packed = (byte) (0x21 + row * 7 + block * 3);
                    int highBits = ((row + block) & 1) == 0 ? 0x5555_aaaa : 0xaaaa_5555;
                    writeQ4_0Block(
                            q4_0.asSlice(row * q4_0RowBytes + block * (long) Q4_0_BLOCK_BYTES, Q4_0_BLOCK_BYTES),
                            scale,
                            packed);
                    writeQ4_1Block(
                            q4_1.asSlice(row * q4_1RowBytes + block * (long) Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                            scale,
                            min,
                            packed);
                    writeQ4_1Block(
                            q4_1NoBias.asSlice(
                                    row * q4_1RowBytes + block * (long) Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                            scale,
                            (short) 0,
                            packed);
                    writeQ5_0Block(
                            q5_0.asSlice(row * q5_0RowBytes + block * (long) Q5_0_BLOCK_BYTES, Q5_0_BLOCK_BYTES),
                            scale,
                            highBits,
                            packed);
                    writeQ5_1Block(
                            q5_1.asSlice(row * q5_1RowBytes + block * (long) Q5_1_BLOCK_BYTES, Q5_1_BLOCK_BYTES),
                            scale,
                            min,
                            highBits,
                            packed);
                    writeQ5_1Block(
                            q5_1NoBias.asSlice(
                                    row * q5_1RowBytes + block * (long) Q5_1_BLOCK_BYTES, Q5_1_BLOCK_BYTES),
                            scale,
                            (short) 0,
                            highBits,
                            packed);
                }
            }

            float[] vector = patternedVector(columns);
            float[] vectorBlockSums =
                    GgufSum.vector32GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());

            assertRawQ4_0Rows(q4_0, rows, columns, q4_0RowBytes, vector);
            assertRawQ4_1Rows(q4_1, rows, columns, q4_1RowBytes, vector, vectorBlockSums);
            assertRawQ4_1NoBiasRows(q4_1NoBias, rows, columns, q4_1RowBytes, vector);
            assertRawQ5_0Rows(q5_0, rows, columns, q5_0RowBytes, vector);
            assertRawQ5_1Rows(q5_1, rows, columns, q5_1RowBytes, vector, vectorBlockSums);
            assertRawQ5_1NoBiasRows(q5_1NoBias, rows, columns, q5_1RowBytes, vector);
        }
    }

    @Test
    void rawQ32TinyBiasedMatVecSlicesUseDirectDots() {
        int rows = 3;
        int blocks = 3;
        int columns = blocks * Q4_0_BLOCK_SIZE;
        long q4RowBytes = blocks * (long) Q4_1_BLOCK_BYTES;
        long q5RowBytes = blocks * (long) Q5_1_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4 = arena.allocate(rows * q4RowBytes);
            MemorySegment q5 = arena.allocate(rows * q5RowBytes);
            for (int row = 0; row < rows; row++) {
                for (int block = 0; block < blocks; block++) {
                    short scale = (short) (0x3800 + (((row + block) & 3) << 9));
                    short min = (short) (0x3000 + (((row + block) % 5) << 7));
                    byte packed = (byte) (0x21 + row * 11 + block * 5);
                    int highBits = ((row + block) & 1) == 0 ? 0x5555_aaaa : 0xaaaa_5555;
                    writeQ4_1Block(
                            q4.asSlice(row * q4RowBytes + block * (long) Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                            scale,
                            min,
                            packed);
                    writeQ5_1Block(
                            q5.asSlice(row * q5RowBytes + block * (long) Q5_1_BLOCK_BYTES, Q5_1_BLOCK_BYTES),
                            scale,
                            min,
                            highBits,
                            packed);
                }
            }

            float[] vector = patternedVector(columns);
            assertTinyDirectQ4_1(q4, rows, columns, q4RowBytes, vector);
            assertTinyDirectQ5_1(q5, rows, columns, q5RowBytes, vector);
        }
    }

    private static float[] patternedVector(int length) {
        float[] vector = new float[length];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 23 - 11) * 0.0625f;
        }
        return vector;
    }

    private static float referenceQ4_0(float[] scales, byte[] packedQuants, float[] vector) {
        float sum = 0.0f;
        int vectorBase = 0;
        for (int block = 0; block < scales.length; block++) {
            int packed = packedQuants[block] & 0xff;
            int low = packed & 0x0f;
            int high = packed >>> 4;
            for (int index = 0; index < 16; index++) {
                sum += scales[block] * (low - 8) * vector[vectorBase + index];
                sum += scales[block] * (high - 8) * vector[vectorBase + 16 + index];
            }
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static float referenceQ5_0(float[] scales, byte[] packedQuants, int[] highBits, float[] vector) {
        float sum = 0.0f;
        int vectorBase = 0;
        for (int block = 0; block < scales.length; block++) {
            int packed = packedQuants[block] & 0xff;
            int low = packed & 0x0f;
            int high = packed >>> 4;
            for (int index = 0; index < 16; index++) {
                int lowValue = low | (((highBits[block] >>> index) & 1) << 4);
                int highValue = high | (((highBits[block] >>> (index + 16)) & 1) << 4);
                sum += scales[block] * (lowValue - 16) * vector[vectorBase + index];
                sum += scales[block] * (highValue - 16) * vector[vectorBase + 16 + index];
            }
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static void assertRawQ4_0Rows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ32RawDot.dotRowQ4_0(segment, row * rowBytes, columns, vector, 0);
        }
        GgufQ32RawRows.fillMatVecRowsQ4_0(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ4_1Rows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorBlockSums) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ32RawDot.dotRowQ4_1(
                    segment, row * rowBytes, columns, vector, vectorBlockSums);
        }
        GgufQ32RawRows.fillMatVecRowsQ4_1(segment, columns, rowBytes, vector, vectorBlockSums, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ4_1NoBiasRows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ32RawDot.dotRowQ4_1NoBias(
                    segment, row * rowBytes, columns, vector, 0);
        }
        GgufQ32RawRows.fillMatVecRowsQ4_1NoBias(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ5_0Rows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ32RawDot.dotRowQ5_0(segment, row * rowBytes, columns, vector, 0);
        }
        GgufQ32RawRows.fillMatVecRowsQ5_0(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ5_1Rows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorBlockSums) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ32RawDot.dotRowQ5_1(
                    segment, row * rowBytes, columns, vector, vectorBlockSums);
        }
        GgufQ32RawRows.fillMatVecRowsQ5_1(segment, columns, rowBytes, vector, vectorBlockSums, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ5_1NoBiasRows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ32RawDot.dotRowQ5_1NoBias(
                    segment, row * rowBytes, columns, vector, 0);
        }
        GgufQ32RawRows.fillMatVecRowsQ5_1NoBias(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyDirectQ4_1(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufRawDot.q4_1(segment, row * rowBytes, columns, vector, true);
        }
        GgufRawMatVec.q4_1(segment, columns, rowBytes, vector, actual, rows, true, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyDirectQ5_1(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufRawDot.q5_1(segment, row * rowBytes, columns, vector, true);
        }
        GgufRawMatVec.q5_1(segment, columns, rowBytes, vector, actual, rows, true, true);
        assertArrayEquals(expected, actual, 0.0f);
    }
}
