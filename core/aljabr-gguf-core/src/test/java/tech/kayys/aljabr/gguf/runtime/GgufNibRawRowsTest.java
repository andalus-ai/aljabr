package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeIQ4NLBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeIQ4XSBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeMXFP4Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeNVFP4Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_NL_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_NL_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_XS_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.MXFP4_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.MXFP4_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;

class GgufNibRawRowsTest {
    private static final int BLOCKS = 5;
    private static final int START_ROW = 1;
    private static final int ROWS_TO_FILL = 7;
    private static final int TOTAL_ROWS = START_ROW + ROWS_TO_FILL;

    @Test
    void mxfp4RowsHandleUnrolledRowsAndTail() {
        assertRows(
                BLOCKS * MXFP4_BLOCK_SIZE,
                MXFP4_BLOCK_BYTES,
                (block, row, index) -> writeMXFP4Block(
                        block,
                        (byte) (126 + ((row + index) & 3)),
                        quant(row, index)),
                GgufNibRawRows::fillMatVecRowsMXFP4,
                (data, rowOffset, columns, vector) ->
                        GgufNibRawDot.dotRowMXFP4(data, rowOffset, columns, vector, 0));
    }

    @Test
    void nvfp4RowsHandleUnrolledRowsAndTail() {
        assertRows(
                BLOCKS * NVFP4_BLOCK_SIZE,
                NVFP4_BLOCK_BYTES,
                (block, row, index) -> writeNVFP4Block(
                        block,
                        (byte) 0x40,
                        (byte) 0x48,
                        (byte) 0x38,
                        (byte) 0x30,
                        quant(row, index)),
                GgufNibRawRows::fillMatVecRowsNVFP4,
                (data, rowOffset, columns, vector) ->
                        GgufNibRawDot.dotRowNVFP4(data, rowOffset, columns, vector, 0));
    }

    @Test
    void iq4NlRowsHandleUnrolledRowsAndTail() {
        assertRows(
                BLOCKS * IQ4_NL_BLOCK_SIZE,
                IQ4_NL_BLOCK_BYTES,
                (block, row, index) -> writeIQ4NLBlock(block, scale(row, index), quant(row, index)),
                GgufNibRawRows::fillMatVecRowsIQ4NL,
                (data, rowOffset, columns, vector) ->
                        GgufNibRawDot.dotRowIQ4NL(data, rowOffset, columns, vector, 0));
    }

    @Test
    void iq4XsRowsHandleUnrolledRowsAndTail() {
        assertRows(
                BLOCKS * QK_K,
                IQ4_XS_BLOCK_BYTES,
                (block, row, index) -> writeIQ4XSBlock(block, scale(row, index), quant(row, index)),
                GgufNibRawRows::fillMatVecRowsIQ4XS,
                (data, rowOffset, columns, vector) ->
                        GgufNibRawDot.dotRowIQ4XS(data, rowOffset, columns, vector, 0));
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
        return (byte) (0x31 + row * 17 + block * 13);
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
            values[index] = (index % 23 - 11) * 0.01875f;
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
