package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufDenseTest.LE_FLOAT;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufHalfFx.LE_SHORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;

class GgufDenseRawTest {
    @Test
    void rawDenseMatVecUsesDirectDotPathForSingleDecodeRow() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = 8;
            float[] vector = ramp(columns);

            MemorySegment f32Segment = arena.allocate((long) columns * Float.BYTES);
            MemorySegment f16Segment = arena.allocate((long) columns * Short.BYTES);
            MemorySegment bf16Segment = arena.allocate((long) columns * Short.BYTES);
            for (int i = 0; i < columns; i++) {
                f32Segment.set(LE_FLOAT, i * (long) Float.BYTES, i + 1.0f);
                f16Segment.set(LE_SHORT, i * (long) Short.BYTES, (short) 0x3c00);
                bf16Segment.set(LE_SHORT, i * (long) Short.BYTES, (short) 0x4000);
            }

            assertSingleRowDenseDot(f32Segment, 0, columns, vector, 204.0f);
            assertSingleRowDenseDot(f16Segment, 1, columns, vector, 36.0f);
            assertSingleRowDenseDot(bf16Segment, 30, columns, vector, 72.0f);
        }
    }

    @Test
    void rawFallbackUsesDirectDotPathForSingleDecodeRow() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = 8;
            MemorySegment segment = arena.allocate((long) columns * Float.BYTES);
            for (int i = 0; i < columns; i++) {
                segment.set(LE_FLOAT, i * (long) Float.BYTES, i + 1.0f);
            }

            float[] vector = ramp(columns);
            float[] output = new float[1];
            GgufRawMatVec.fallback(segment, 0, columns, (long) columns * Float.BYTES, vector, output, 1, true);

            assertEquals(204.0f, output[0], 0.0f);
        }
    }

    @Test
    void rawDenseTinyMatVecSlicesBypassWorkerProbe() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        GgufRows.clearRawWorkerAccessCache();
        int rows = 4;
        int columns = 8;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment f32Segment = arena.allocate(rows * (long) columns * Float.BYTES);
            MemorySegment f16Segment = arena.allocate(rows * (long) columns * Short.BYTES);
            MemorySegment bf16Segment = arena.allocate(rows * (long) columns * Short.BYTES);
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float value = (row + 1) * 0.375f + (column - 3) * 0.125f;
                    f32Segment.set(LE_FLOAT, ((long) row * columns + column) * Float.BYTES, value);
                    f16Segment.set(LE_SHORT, ((long) row * columns + column) * Short.BYTES,
                            Float.floatToFloat16(value));
                    bf16Segment.set(LE_SHORT, ((long) row * columns + column) * Short.BYTES, f32ToBf16(value));
                }
            }

            float[] vector = patternedVector(columns);
            assertTinyDenseRows(
                    f32Segment,
                    GgufDenseRoute.F32,
                    GgmlType.F32.id,
                    (long) columns * Float.BYTES,
                    columns,
                    vector,
                    GgufDenseRows::fillMatVecRowsF32);
            assertTinyDenseRows(
                    f16Segment,
                    GgufDenseRoute.F16,
                    GgmlType.F16.id,
                    (long) columns * Short.BYTES,
                    columns,
                    vector,
                    GgufDenseRows::fillMatVecRowsF16);
            assertTinyDenseRows(
                    bf16Segment,
                    GgufDenseRoute.BF16,
                    GgmlType.BF16.id,
                    (long) columns * Short.BYTES,
                    columns,
                    vector,
                    GgufDenseRows::fillMatVecRowsBF16);
            assertTinyFallbackRows(
                    f32Segment,
                    GgmlType.F32.id,
                    (long) columns * Float.BYTES,
                    columns,
                    vector);

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
    void rawDenseRowWalkersHandleUnrolledRowsAndTail() {
        try (Arena arena = Arena.ofConfined()) {
            int rows = 7;
            int columns = GgufTensorOps.preferredFloatVectorLanes() * 2 + 3;
            float[] vector = patternedVector(columns);
            MemorySegment f32Segment = arena.allocate((long) rows * columns * Float.BYTES);
            MemorySegment f16Segment = arena.allocate((long) rows * columns * Short.BYTES);
            MemorySegment bf16Segment = arena.allocate((long) rows * columns * Short.BYTES);
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float value = (row + 1) * 0.25f + (column % 11 - 5) * 0.125f;
                    f32Segment.set(LE_FLOAT, ((long) row * columns + column) * Float.BYTES, value);
                    f16Segment.set(LE_SHORT, ((long) row * columns + column) * Short.BYTES,
                            Float.floatToFloat16(value));
                    bf16Segment.set(LE_SHORT, ((long) row * columns + column) * Short.BYTES, f32ToBf16(value));
                }
            }

            assertDenseRows(
                    rows,
                    columns,
                    f32Segment,
                    (long) columns * Float.BYTES,
                    vector,
                    (data, rowOffset) -> GgufDenseDot.dotRowF32(data, rowOffset, columns, vector, 0),
                    GgufDenseRows::fillMatVecRowsF32);
            assertDenseRows(
                    rows,
                    columns,
                    f32Segment,
                    (long) columns * Float.BYTES,
                    vector,
                    (data, rowOffset) -> GgufDenseDot.dotRowF32Scalar(data, rowOffset, columns, vector, 0),
                    GgufDenseRows::fillMatVecRowsF32Scalar);
            assertDenseRows(
                    rows,
                    columns,
                    f16Segment,
                    (long) columns * Short.BYTES,
                    vector,
                    (data, rowOffset) -> GgufDenseDot.dotRowF16(data, rowOffset, columns, vector, 0),
                    GgufDenseRows::fillMatVecRowsF16);
            assertDenseRows(
                    rows,
                    columns,
                    bf16Segment,
                    (long) columns * Short.BYTES,
                    vector,
                    (data, rowOffset) -> GgufDenseDot.dotRowBF16(data, rowOffset, columns, vector, 0),
                    GgufDenseRows::fillMatVecRowsBF16);
            assertDenseRows(
                    rows,
                    columns,
                    bf16Segment,
                    (long) columns * Short.BYTES,
                    vector,
                    (data, rowOffset) -> GgufDenseDot.dotRowBF16Scalar(data, rowOffset, columns, vector, 0),
                    GgufDenseRows::fillMatVecRowsBF16Scalar);
        }
    }

    @Test
    void rawFallbackRowWalkerHandlesUnrolledRowsAndTail() {
        try (Arena arena = Arena.ofConfined()) {
            int startRow = 1;
            int rows = 8;
            int columns = 9;
            long rowBytes = (long) columns * Float.BYTES;
            MemorySegment segment = arena.allocate(rows * rowBytes);
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float value = (row + 1) * 0.5f + (column - 4) * 0.125f;
                    segment.set(LE_FLOAT, row * rowBytes + column * (long) Float.BYTES, value);
                }
            }

            float[] vector = patternedVector(columns);
            float[] output = new float[rows];
            Arrays.fill(output, -999.0f);

            GgufRawRows.fillFallback(segment, 0, columns, rowBytes, vector, output, startRow, rows);

            assertEquals(-999.0f, output[0], 0.0f);
            for (int row = startRow; row < rows; row++) {
                float expected = GgufRowDot.row(segment, row * rowBytes, 0, columns, vector, 0);
                assertEquals(expected, output[row], 0.0f);
            }
        }
    }

    private static void assertSingleRowDenseDot(
            MemorySegment segment,
            int typeId,
            int columns,
            float[] vector,
            float expected) {
        GGUFTensorInfo tensor = new GGUFTensorInfo(
                "dense.single." + typeId, new long[]{columns, 1}, typeId, 0, segment.byteSize());
        GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
        float[] output = new float[1];

        GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

        assertEquals(expected, output[0], 0.0f);
    }

    private static void assertTinyDenseRows(
            MemorySegment segment,
            int route,
            int typeId,
            long rowBytes,
            int columns,
            float[] vector,
            GgufRawRows.RawRowFiller filler) {
        int rows = 4;
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        filler.fill(segment, columns, rowBytes, vector, expected, 0, rows);
        GgufRawMatVec.denseRaw(segment, route, typeId, columns, rowBytes, vector, actual, rows, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyFallbackRows(
            MemorySegment segment,
            int typeId,
            long rowBytes,
            int columns,
            float[] vector) {
        int rows = 4;
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufRawRows.fillFallback(segment, typeId, columns, rowBytes, vector, expected, 0, rows);
        GgufRawMatVec.fallback(segment, typeId, columns, rowBytes, vector, actual, rows, true);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertDenseRows(
            int rows,
            int columns,
            MemorySegment segment,
            long rowBytes,
            float[] vector,
            DenseDot dot,
            DenseFiller filler) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        long rowOffset = 0L;
        for (int row = 0; row < rows; row++) {
            expected[row] = dot.dot(segment, rowOffset);
            rowOffset += rowBytes;
        }

        filler.fill(segment, columns, rowBytes, vector, actual, 0, rows);

        assertArrayEquals(expected, actual, 0.0f);
    }

    private static float[] patternedVector(int columns) {
        float[] vector = new float[columns];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 13 - 6) * 0.0625f;
        }
        return vector;
    }

    private static short f32ToBf16(float value) {
        return (short) (Float.floatToRawIntBits(value) >>> 16);
    }

    @FunctionalInterface
    private interface DenseDot {
        float dot(MemorySegment data, long rowOffset);
    }

    @FunctionalInterface
    private interface DenseFiller {
        void fill(
                MemorySegment data,
                int columns,
                long rowBytes,
                float[] vector,
                float[] output,
                int startRow,
                int endRow);
    }
}
