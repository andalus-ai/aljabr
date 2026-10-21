package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ1_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ1_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ2_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q1_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q1_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ2_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ2_0_BLOCK_SIZE;

class GgufTqRawRowsTest {
    private static final int BLOCKS = 5;
    private static final int START_ROW = 1;
    private static final int ROWS_TO_FILL = 7;
    private static final int TOTAL_ROWS = START_ROW + ROWS_TO_FILL;

    @Test
    void q1RowsHandleUnrolledRowsAndTail() {
        assertRows(
                BLOCKS * Q1_0_BLOCK_SIZE,
                Q1_0_BLOCK_BYTES,
                (block, row, index) -> writeQ1_0Block(block, scale(row, index), quant(row, index)),
                GgufTqRawRows::fillMatVecRowsQ1_0,
                (data, rowOffset, columns, vector) ->
                        GgufTqRawDot.dotRowQ1_0(data, rowOffset, columns, vector, 0));
    }

    @Test
    void tq1RowsHandleUnrolledRowsAndTail() {
        assertRows(
                BLOCKS * TQ1_0_BLOCK_SIZE,
                TQ1_0_BLOCK_BYTES,
                (block, row, index) -> writeTQ1_0Block(block, scale(row, index), quant(row, index)),
                GgufTqRawRows::fillMatVecRowsTQ1_0,
                (data, rowOffset, columns, vector) ->
                        GgufTqRawDot.dotRowTQ1_0(data, rowOffset, columns, vector, 0));
    }

    @Test
    void tq2RowsHandleUnrolledRowsAndTail() {
        assertRows(
                BLOCKS * TQ2_0_BLOCK_SIZE,
                TQ2_0_BLOCK_BYTES,
                (block, row, index) -> writeTQ2_0Block(block, scale(row, index), quant(row, index)),
                GgufTqRawRows::fillMatVecRowsTQ2_0,
                (data, rowOffset, columns, vector) ->
                        GgufTqRawDot.dotRowTQ2_0(data, rowOffset, columns, vector, 0));
    }

    private static void assertRows(
            int columns,
            int blockBytes,
            BlockWriter writer,
            RowFiller filler,
            RowDot dot) {
        long rowBytes = BLOCKS * (long) blockBytes;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment data = arena.allocate(TOTAL_ROWS * rowBytes);
            for (int row = 0; row < TOTAL_ROWS; row++) {
                long rowOffset = row * rowBytes;
                for (int block = 0; block < BLOCKS; block++) {
                    writer.write(
                            data.asSlice(rowOffset + block * (long) blockBytes, blockBytes),
                            row,
                            block);
                }
            }

            float[] vector = wave(columns);
            float[] output = new float[TOTAL_ROWS];
            Arrays.fill(output, -999.0f);

            filler.fill(data, columns, rowBytes, vector, output, START_ROW, TOTAL_ROWS);

            assertEquals(-999.0f, output[0], 0.0f);
            for (int row = START_ROW; row < TOTAL_ROWS; row++) {
                float expected = dot.dot(data, row * rowBytes, columns, vector);
                assertClose(expected, output[row]);
            }
        }
    }

    private static byte quant(int row, int block) {
        return (byte) (0x21 + row * 11 + block * 7);
    }

    private static short scale(int row, int block) {
        return switch ((row + block) % 3) {
            case 0 -> (short) 0x3c00;
            case 1 -> (short) 0x4000;
            default -> (short) 0x3800;
        };
    }

    private static float[] wave(int length) {
        float[] values = new float[length];
        for (int index = 0; index < values.length; index++) {
            values[index] = (index % 19 - 9) * 0.021875f;
        }
        return values;
    }

    private static void assertClose(float expected, float actual) {
        float tolerance = Math.max(1.0e-3f, Math.abs(expected) * 1.0e-5f);
        assertEquals(expected, actual, tolerance);
    }

    private interface BlockWriter {
        void write(MemorySegment block, int row, int index);
    }

    private interface RowFiller {
        void fill(
                MemorySegment data,
                int columns,
                long rowBytes,
                float[] vector,
                float[] output,
                int startRow,
                int endRow);
    }

    private interface RowDot {
        float dot(MemorySegment data, long rowOffset, int columns, float[] vector);
    }
}
