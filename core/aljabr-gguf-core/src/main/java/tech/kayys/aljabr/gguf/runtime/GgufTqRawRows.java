package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufTqRawDot.*;

import java.lang.foreign.MemorySegment;

/**
 * Row walkers for raw Q1 and ternary GGUF matrices.
 *
 * <p>This helper advances matrix rows for Q1_0, TQ1_0, and TQ2_0 while
 * delegating the row arithmetic to {@link GgufTqRawDot}.</p>
 * @author bhangun
 */
final class GgufTqRawRows {
    private GgufTqRawRows() {
    }

    static void fillMatVecRowsQ1_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        long fourRowBytes = rowBytes << 2;
        for (; row < tailStart; row += 4) {
            dotRowsQ1_0_4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ1_0(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsTQ1_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        long fourRowBytes = rowBytes << 2;
        for (; row < tailStart; row += 4) {
            dotRowsTQ1_0_4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowTQ1_0(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsTQ2_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        long fourRowBytes = rowBytes << 2;
        for (; row < tailStart; row += 4) {
            dotRowsTQ2_0_4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowTQ2_0(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }
}
