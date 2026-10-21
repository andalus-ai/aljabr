package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ8RawTest {
    private static final int VECTOR_OFFSET = 5;

    @Test
    void rawQ8MatVecSkipsPreparedAdmissionWhenCacheIsDisabled() {
        String previousMinRows = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q8.cache_max_bytes");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q8.cache_max_bytes", "0");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 34);
            writeQ8Block(segment.asSlice(0, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(34, 34), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8.raw.disabled_cache", new long[]{32, 2}, 8, 0, 2L * 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(32), output, 2, true);

            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(64.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q8.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ8MatVecStreamsWhenPreparedCacheIsTooSmall() {
        String previousMinRows = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q8.cache_max_bytes");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q8.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 34);
            writeQ8Block(segment.asSlice(0, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(34, 34), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8.raw.cache_miss", new long[]{32, 2}, 8, 0, 2L * 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(32), output, 2, true);

            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(64.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q8.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ8MatVecPreservesSignedVectorLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q8.cache_max_bytes");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q8.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 34);
            writeQ8RampBlock(segment.asSlice(0, 34), (short) 0x3c00, -16);
            writeQ8RampBlock(segment.asSlice(34, 34), (short) 0x3c00, 16);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8.raw.lanes", new long[]{32, 2}, 8, 0, 2L * 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ramp(32), output, 2, true);

            assertEquals(2464.0f, output[0], 0.0f);
            assertEquals(-2464.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q8MatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q8.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ8_1MatVecSkipsPreparedAdmissionWhenCacheIsDisabled() {
        String previousMinRows = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q8.cache_max_bytes");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q8.cache_max_bytes", "0");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 36);
            writeQ8_1Block(segment.asSlice(0, 36), (short) 0x3c00, (byte) 1);
            writeQ8_1Block(segment.asSlice(36, 36), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8_1.raw.disabled_cache", new long[]{32, 2}, 9, 0, 2L * 36);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(32), output, 2, true);

            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(64.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q8.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ8FamilyTinyMatVecSlicesBypassWorkerProbe() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        GgufRows.clearRawWorkerAccessCache();
        try {
            assertTinyQ8Route(
                    GgufQ8Route.Q1_0,
                    GgmlType.Q1_0.id,
                    2,
                    GgufQuantFormats.Q1_0_BLOCK_SIZE,
                    GgufQuantFormats.Q1_0_BLOCK_BYTES,
                    (block, row, index) -> writeQ1_0Block(block, (short) 0x3c00, q8Pattern(row, index)),
                    GgufTqRawRows::fillMatVecRowsQ1_0);
            assertTinyQ8Route(
                    GgufQ8Route.TQ1_0,
                    GgmlType.TQ1_0.id,
                    2,
                    GgufQuantFormats.TQ1_0_BLOCK_SIZE,
                    GgufQuantFormats.TQ1_0_BLOCK_BYTES,
                    (block, row, index) -> writeTQ1_0Block(block, (short) 0x3c00, q8Pattern(row, index)),
                    GgufTqRawRows::fillMatVecRowsTQ1_0);
            assertTinyQ8Route(
                    GgufQ8Route.TQ2_0,
                    GgmlType.TQ2_0.id,
                    2,
                    GgufQuantFormats.TQ2_0_BLOCK_SIZE,
                    GgufQuantFormats.TQ2_0_BLOCK_BYTES,
                    (block, row, index) -> writeTQ2_0Block(block, (short) 0x3c00, q8Pattern(row, index)),
                    GgufTqRawRows::fillMatVecRowsTQ2_0);
            assertTinyQ8Route(
                    GgufQ8Route.MXFP4,
                    GgmlType.MXFP4.id,
                    2,
                    GgufQuantFormats.MXFP4_BLOCK_SIZE,
                    GgufQuantFormats.MXFP4_BLOCK_BYTES,
                    (block, row, index) -> writeMXFP4Block(
                            block, (byte) (126 + ((row + index) & 3)), q8Pattern(row, index)),
                    GgufNibRawRows::fillMatVecRowsMXFP4);
            assertTinyQ8Route(
                    GgufQ8Route.NVFP4,
                    GgmlType.NVFP4.id,
                    2,
                    GgufQuantFormats.NVFP4_BLOCK_SIZE,
                    GgufQuantFormats.NVFP4_BLOCK_BYTES,
                    (block, row, index) -> writeNVFP4Block(
                            block, (byte) 0x40, (byte) 0x48, (byte) 0x38, (byte) 0x30, q8Pattern(row, index)),
                    GgufNibRawRows::fillMatVecRowsNVFP4);
            assertTinyQ8Route(
                    GgufQ8Route.Q8_0,
                    GgmlType.Q8_0.id,
                    2,
                    Q8_0_BLOCK_SIZE,
                    Q8_0_BLOCK_BYTES,
                    (block, row, index) -> writePatternQ8Block(block, (short) 0x3c00, 17 + row * 11 + index * 5),
                    GgufQ8RawRows::fillMatVecRowsQ8_0);
            assertTinyQ8Route(
                    GgufQ8Route.Q8_1,
                    GgmlType.Q8_1.id,
                    2,
                    GgufQuantFormats.Q8_1_BLOCK_SIZE,
                    Q8_1_BLOCK_BYTES,
                    (block, row, index) -> writePatternQ8_1Block(block, (short) 0x3c00, 19 + row * 13 + index * 7),
                    GgufQ8RawRows::fillMatVecRowsQ8_1);
            assertTinyQ8Route(
                    GgufQ8Route.Q8_K,
                    GgmlType.Q8_K.id,
                    2,
                    QK_K,
                    Q8_K_BLOCK_BYTES,
                    (block, row, index) -> writePatternQ8KBlock(block, 1.0f + row * 0.25f, 23 + row * 17 + index * 3),
                    GgufQ8RawRows::fillMatVecRowsQ8K);
            assertTinyQ8Route(
                    GgufQ8Route.IQ4_NL,
                    GgmlType.IQ4_NL.id,
                    2,
                    GgufQuantFormats.IQ4_NL_BLOCK_SIZE,
                    GgufQuantFormats.IQ4_NL_BLOCK_BYTES,
                    (block, row, index) -> writeIQ4NLBlock(block, (short) 0x3c00, q8Pattern(row, index)),
                    GgufNibRawRows::fillMatVecRowsIQ4NL);
            assertTinyQ8Route(
                    GgufQ8Route.IQ4_XS,
                    GgmlType.IQ4_XS.id,
                    2,
                    QK_K,
                    GgufQuantFormats.IQ4_XS_BLOCK_BYTES,
                    (block, row, index) -> writeIQ4XSBlock(block, (short) 0x3c00, q8Pattern(row, index)),
                    GgufNibRawRows::fillMatVecRowsIQ4XS);

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
    void rawQ8_0VectorAccumulatorMatchesIndependentReference() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(5L * Q8_0_BLOCK_BYTES);
            short[] halfScales = {(short) 0x3c00, (short) 0x4000, (short) 0xbc00, (short) 0x4200, (short) 0x3800};
            float[] scales = {1.0f, 2.0f, -1.0f, 3.0f, 0.5f};
            int[] seeds = {11, 29, 47, 83, 109};
            for (int block = 0; block < scales.length; block++) {
                writePatternQ8Block(
                        segment.asSlice(block * (long) Q8_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES),
                        halfScales[block],
                        seeds[block]);
            }

            int columns = scales.length * Q8_0_BLOCK_SIZE;
            float[] vector = patternedVector(columns + VECTOR_OFFSET);
            float expected = referenceRawQ8(scales, seeds, vector, VECTOR_OFFSET);
            float scalar = GgufQ8RawDot.dotRowQ8_0Scalar(segment, 0, columns, vector, VECTOR_OFFSET);
            float vectorized = GgufQ8RawDot.dotRowQ8_0Vector(segment, 0, columns, vector, VECTOR_OFFSET);

            assertEquals(expected, scalar, 1.0e-3f);
            assertEquals(expected, vectorized, 1.0e-3f);
        }
    }

    @Test
    void rawQ8_1VectorAccumulatorMatchesIndependentReference() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(5L * Q8_1_BLOCK_BYTES);
            short[] halfScales = {(short) 0x3c00, (short) 0x4000, (short) 0xbc00, (short) 0x4200, (short) 0x3800};
            float[] scales = {1.0f, 2.0f, -1.0f, 3.0f, 0.5f};
            int[] seeds = {13, 31, 53, 89, 127};
            for (int block = 0; block < scales.length; block++) {
                writePatternQ8_1Block(
                        segment.asSlice(block * (long) Q8_1_BLOCK_BYTES, Q8_1_BLOCK_BYTES),
                        halfScales[block],
                        seeds[block]);
            }

            int columns = scales.length * Q8_0_BLOCK_SIZE;
            float[] vector = patternedVector(columns + VECTOR_OFFSET);
            float expected = referenceRawQ8(scales, seeds, vector, VECTOR_OFFSET);
            float scalar = GgufQ8RawDot.dotRowQ8_1Scalar(segment, 0, columns, vector, VECTOR_OFFSET);
            float vectorized = GgufQ8RawDot.dotRowQ8_1Vector(segment, 0, columns, vector, VECTOR_OFFSET);

            assertEquals(expected, scalar, 1.0e-3f);
            assertEquals(expected, vectorized, 1.0e-3f);
        }
    }

    @Test
    void rawQ8RowWalkersHandleUnrolledRowsAndTail() {
        int rows = 7;
        int q8Blocks = 5;
        int q8Columns = q8Blocks * Q8_0_BLOCK_SIZE;
        int q8KBlocks = 5;
        int q8KColumns = q8KBlocks * QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q8_0 = arena.allocate(rows * q8Blocks * (long) Q8_0_BLOCK_BYTES);
            MemorySegment q8_1 = arena.allocate(rows * q8Blocks * (long) Q8_1_BLOCK_BYTES);
            MemorySegment q8K = arena.allocate(rows * q8KBlocks * (long) Q8_K_BLOCK_BYTES);
            for (int row = 0; row < rows; row++) {
                for (int block = 0; block < q8Blocks; block++) {
                    short scale = (short) (0x3800 + (((row + block) & 3) << 10));
                    writePatternQ8Block(
                            q8_0.asSlice(
                                    (row * q8Blocks + block) * (long) Q8_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES),
                            scale,
                            17 + row * 19 + block * 7);
                    writePatternQ8_1Block(
                            q8_1.asSlice(
                                    (row * q8Blocks + block) * (long) Q8_1_BLOCK_BYTES, Q8_1_BLOCK_BYTES),
                            scale,
                            23 + row * 29 + block * 11);
                }
                for (int block = 0; block < q8KBlocks; block++) {
                    writePatternQ8KBlock(
                            q8K.asSlice(
                                    (row * q8KBlocks + block) * (long) Q8_K_BLOCK_BYTES, Q8_K_BLOCK_BYTES),
                            0.5f + row * 0.25f + block * 0.125f,
                            31 + row * 37 + block * 13);
                }
            }

            float[] vector32 = patternedVector(q8Columns);
            float[] vectorK = patternedVector(q8KColumns);

            assertRawQ8_0Rows(q8_0, rows, q8Columns, q8Blocks * (long) Q8_0_BLOCK_BYTES, vector32);
            assertRawQ8_0ScalarRows(q8_0, rows, q8Columns, q8Blocks * (long) Q8_0_BLOCK_BYTES, vector32);
            assertRawQ8_1Rows(q8_1, rows, q8Columns, q8Blocks * (long) Q8_1_BLOCK_BYTES, vector32);
            assertRawQ8_1ScalarRows(q8_1, rows, q8Columns, q8Blocks * (long) Q8_1_BLOCK_BYTES, vector32);
            assertRawQ8KRows(q8K, rows, q8KColumns, q8KBlocks * (long) Q8_K_BLOCK_BYTES, vectorK);
            assertRawQ8KScalarRows(q8K, rows, q8KColumns, q8KBlocks * (long) Q8_K_BLOCK_BYTES, vectorK);
        }
    }

    private static void assertTinyQ8Route(
            int route,
            int typeId,
            int blocks,
            int blockSize,
            int blockBytes,
            BlockWriter writer,
            GgufRawRows.RawRowFiller filler) {
        int rows = 4;
        int columns = blocks * blockSize;
        long rowBytes = blocks * (long) blockBytes;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rows * rowBytes);
            for (int row = 0; row < rows; row++) {
                long rowOffset = row * rowBytes;
                for (int block = 0; block < blocks; block++) {
                    writer.write(segment.asSlice(rowOffset + block * (long) blockBytes, blockBytes), row, block);
                }
            }

            float[] vector = patternedVector(columns);
            float[] expected = new float[rows];
            float[] actual = new float[rows];
            filler.fill(segment, columns, rowBytes, vector, expected, 0, rows);
            GgufRawMatVec.q8Raw(segment, route, typeId, columns, rowBytes, vector, actual, rows, true);

            assertArrayEquals(expected, actual, 0.0f);
        }
    }

    private static void writePatternQ8Block(MemorySegment block, short halfScale, int seed) {
        writeQ8Block(block, halfScale, (byte) 0);
        writePatternQuants(block, Short.BYTES, seed);
    }

    private static void writePatternQ8_1Block(MemorySegment block, short halfScale, int seed) {
        writeQ8_1Block(block, halfScale, (byte) 0);
        writePatternQuants(block, 2 * Short.BYTES, seed);
    }

    private static void writePatternQ8KBlock(MemorySegment block, float scale, int seed) {
        writeQ8KBlock(block, scale, (byte) 0);
        for (int index = 0; index < QK_K; index++) {
            block.set(ValueLayout.JAVA_BYTE, Float.BYTES + index, q8Pattern(seed, index));
        }
    }

    private static void writePatternQuants(MemorySegment block, long quantOffset, int seed) {
        for (int index = 0; index < Q8_0_BLOCK_SIZE; index++) {
            block.set(ValueLayout.JAVA_BYTE, quantOffset + index, q8Pattern(seed, index));
        }
    }

    private static float[] patternedVector(int length) {
        float[] vector = new float[length];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 19 - 9) * 0.0625f;
        }
        return vector;
    }

    private static float referenceRawQ8(float[] scales, int[] seeds, float[] vector) {
        return referenceRawQ8(scales, seeds, vector, 0);
    }

    private static float referenceRawQ8(float[] scales, int[] seeds, float[] vector, int vectorOffset) {
        float sum = 0.0f;
        int vectorBase = vectorOffset;
        for (int block = 0; block < scales.length; block++) {
            for (int index = 0; index < Q8_0_BLOCK_SIZE; index++) {
                sum += scales[block] * q8Pattern(seeds[block], index) * vector[vectorBase + index];
            }
            vectorBase += Q8_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static byte q8Pattern(int seed, int index) {
        return (byte) (seed + index * 41);
    }

    @FunctionalInterface
    private interface BlockWriter {
        void write(MemorySegment block, int row, int index);
    }

    private static void assertRawQ8_0Rows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        long rowOffset = 0L;
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ8RawDot.dotRowQ8_0(segment, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
        GgufQ8RawRows.fillMatVecRowsQ8_0(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ8_0ScalarRows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        long rowOffset = 0L;
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ8RawDot.dotRowQ8_0Scalar(segment, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
        GgufQ8RawRows.fillMatVecRowsQ8_0Scalar(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ8_1Rows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        long rowOffset = 0L;
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ8RawDot.dotRowQ8_1(segment, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
        GgufQ8RawRows.fillMatVecRowsQ8_1(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ8_1ScalarRows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        long rowOffset = 0L;
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ8RawDot.dotRowQ8_1Scalar(segment, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
        GgufQ8RawRows.fillMatVecRowsQ8_1Scalar(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ8KRows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        long rowOffset = 0L;
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ8RawDot.dotRowQ8K(segment, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
        GgufQ8RawRows.fillMatVecRowsQ8K(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertRawQ8KScalarRows(
            MemorySegment segment,
            int rows,
            int columns,
            long rowBytes,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        long rowOffset = 0L;
        for (int row = 0; row < rows; row++) {
            expected[row] = GgufQ8RawDot.dotRowQ8KScalar(segment, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
        GgufQ8RawRows.fillMatVecRowsQ8KScalar(segment, columns, rowBytes, vector, actual, 0, rows);
        assertArrayEquals(expected, actual, 0.0f);
    }
}
