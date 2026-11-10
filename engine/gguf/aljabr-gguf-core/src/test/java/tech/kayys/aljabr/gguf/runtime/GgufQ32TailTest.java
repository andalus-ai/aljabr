package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;

class GgufQ32TailTest {
    @Test
    void rawBiasTailEntrypointsUseDirectDotPathForSingleDecodeRow() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4Segment = arena.allocate(20);
            MemorySegment q5Segment = arena.allocate(24);
            writeQ4_1Block(q4Segment, (short) 0x3c00, (short) 0, (byte) 0x21);
            writeQ5_1Block(q5Segment, (short) 0x3c00, (short) 0, -1, (byte) 0x21);

            float[] vector = ramp(32);
            float[] output = new float[1];
            GgufRawMatVec.q4_1(q4Segment, 32, 20, vector, output, 1, true, false);
            assertEquals(GgufQ32RawDot.dotRowQ4_1NoBias(q4Segment, 0, 32, vector, 0), output[0], 0.0f);

            GgufRawMatVec.q5_1(q5Segment, 32, 24, vector, output, 1, true, false);
            assertEquals(GgufQ32RawDot.dotRowQ5_1NoBias(q5Segment, 0, 32, vector, 0), output[0], 0.0f);
        }
    }

    @Test
    void rawQ32TinyNoBiasMatVecSlicesBypassWorkerProbe() {
        String previousMinOps = System.getProperty("aljabr.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("aljabr.gguf.parallel_threads");
        String previousChunks = System.getProperty("aljabr.gguf.parallel_chunks_per_thread");
        System.setProperty("aljabr.gguf.parallel_min_ops", "1");
        System.setProperty("aljabr.gguf.parallel_threads", "2");
        System.setProperty("aljabr.gguf.parallel_chunks_per_thread", "1");
        GgufParallelConfig.resetParallelConfig();
        GgufRows.clearRawWorkerAccessCache();
        int rows = 4;
        int columns = GgufQuantFormats.Q4_0_BLOCK_SIZE;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4_0 = arena.allocate(rows * (long) GgufQuantFormats.Q4_0_BLOCK_BYTES);
            MemorySegment q4_1 = arena.allocate(rows * (long) GgufQuantFormats.Q4_1_BLOCK_BYTES);
            MemorySegment q5_0 = arena.allocate(rows * (long) GgufQuantFormats.Q5_0_BLOCK_BYTES);
            MemorySegment q5_1 = arena.allocate(rows * (long) GgufQuantFormats.Q5_1_BLOCK_BYTES);
            for (int row = 0; row < rows; row++) {
                writeQ4_0Block(
                        q4_0.asSlice(
                                row * (long) GgufQuantFormats.Q4_0_BLOCK_BYTES,
                                GgufQuantFormats.Q4_0_BLOCK_BYTES),
                        (short) 0x3c00,
                        (byte) (0x21 + row));
                writeQ4_1Block(
                        q4_1.asSlice(
                                row * (long) GgufQuantFormats.Q4_1_BLOCK_BYTES,
                                GgufQuantFormats.Q4_1_BLOCK_BYTES),
                        (short) 0x3c00,
                        (short) 0,
                        (byte) (0x31 + row));
                writeQ5_0Block(
                        q5_0.asSlice(
                                row * (long) GgufQuantFormats.Q5_0_BLOCK_BYTES,
                                GgufQuantFormats.Q5_0_BLOCK_BYTES),
                        (short) 0x3c00,
                        0xFFFF0000 ^ row,
                        (byte) (0x41 + row));
                writeQ5_1Block(
                        q5_1.asSlice(
                                row * (long) GgufQuantFormats.Q5_1_BLOCK_BYTES,
                                GgufQuantFormats.Q5_1_BLOCK_BYTES),
                        (short) 0x3c00,
                        (short) 0,
                        -1,
                        (byte) (0x51 + row));
            }

            float[] vector = ones(columns);
            assertTinyQ4_0(q4_0, rows, columns, vector);
            assertTinyQ4_1NoBias(q4_1, rows, columns, vector);
            assertTinyQ5_0(q5_0, rows, columns, vector);
            assertTinyQ5_1NoBias(q5_1, rows, columns, vector);

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

    private static void assertTinyQ4_0(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufQ32RawRows.fillMatVecRowsQ4_0(
                segment, columns, GgufQuantFormats.Q4_0_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q32Raw(
                segment, GgufQ32Route.Q4_0, 2, columns, GgufQuantFormats.Q4_0_BLOCK_BYTES, vector, actual, rows, true, false);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyQ4_1NoBias(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufQ32RawRows.fillMatVecRowsQ4_1NoBias(
                segment, columns, GgufQuantFormats.Q4_1_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q4_1(segment, columns, GgufQuantFormats.Q4_1_BLOCK_BYTES, vector, actual, rows, true, false);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyQ5_0(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufQ32RawRows.fillMatVecRowsQ5_0(
                segment, columns, GgufQuantFormats.Q5_0_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q32Raw(
                segment, GgufQ32Route.Q5_0, 6, columns, GgufQuantFormats.Q5_0_BLOCK_BYTES, vector, actual, rows, true, false);
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static void assertTinyQ5_1NoBias(
            MemorySegment segment,
            int rows,
            int columns,
            float[] vector) {
        float[] expected = new float[rows];
        float[] actual = new float[rows];
        GgufQ32RawRows.fillMatVecRowsQ5_1NoBias(
                segment, columns, GgufQuantFormats.Q5_1_BLOCK_BYTES, vector, expected, 0, rows);
        GgufRawMatVec.q5_1(segment, columns, GgufQuantFormats.Q5_1_BLOCK_BYTES, vector, actual, rows, true, false);
        assertArrayEquals(expected, actual, 0.0f);
    }
}
