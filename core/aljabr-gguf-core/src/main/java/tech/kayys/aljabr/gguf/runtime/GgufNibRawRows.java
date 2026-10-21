package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufNibRawDot.*;

import java.lang.foreign.MemorySegment;

/**
 * Row walkers for raw nibble-coded GGUF matrices.
 *
 * <p>The methods here only advance row offsets for MXFP4, NVFP4, IQ4_NL, and
 * IQ4_XS. Format arithmetic stays isolated in {@link GgufNibRawDot}.</p>
 * @author bhangun
 */
final class GgufNibRawRows {
    private GgufNibRawRows() {
    }

    static void fillMatVecRowsMXFP4(
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
            dotRowsMXFP4_4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowMXFP4(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsNVFP4(
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
            dotRowsNVFP4_4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowNVFP4(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsIQ4NL(
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
            dotRowsIQ4NL_4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowIQ4NL(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsIQ4XS(
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
            dotRowsIQ4XS_4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowIQ4XS(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }
}
