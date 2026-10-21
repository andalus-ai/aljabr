package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawFourRows.*;

import java.lang.foreign.MemorySegment;

/**
 * Row walkers for raw Q4/Q5 Q32-family matrices.
 *
 * <p>The methods here only advance rows and write mat-vec outputs. Arithmetic
 * stays in {@link GgufQ32RawDot}, keeping traversal separate from quant math.</p>
 * @author bhangun
 */
final class GgufQ32RawRows {
    private GgufQ32RawRows() {
    }

    static void fillMatVecRowsQ4_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            dotRowsQ4_0(data, rowOffset, rowBytes, columns, vector, 0, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ4_0(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4_1(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorBlockSums,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            dotRowsQ4_1(data, rowOffset, rowBytes, columns, vector, vectorBlockSums, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ4_1(data, rowOffset, columns, vector, vectorBlockSums);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4_1Direct(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            fillMatVecRowsQ4_1Direct(
                    data, columns, rowBytes, vector, output, startRow, endRow,
                    cachedHasBlockBiases.booleanValue());
            return;
        }
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q4_1(data, rowOffset, columns, vector, cachedHasBlockBiases);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4_1Direct(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            boolean hasBlockBiases) {
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q4_1(data, rowOffset, columns, vector, hasBlockBiases);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4_1NoBias(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            dotRowsQ4_1NoBias(data, rowOffset, rowBytes, columns, vector, 0, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ4_1NoBias(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            dotRowsQ5_0(data, rowOffset, rowBytes, columns, vector, 0, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ5_0(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5_1(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorBlockSums,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            dotRowsQ5_1(data, rowOffset, rowBytes, columns, vector, vectorBlockSums, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ5_1(data, rowOffset, columns, vector, vectorBlockSums);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5_1Direct(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            fillMatVecRowsQ5_1Direct(
                    data, columns, rowBytes, vector, output, startRow, endRow,
                    cachedHasBlockBiases.booleanValue());
            return;
        }
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q5_1(data, rowOffset, columns, vector, cachedHasBlockBiases);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5_1Direct(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            boolean hasBlockBiases) {
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q5_1(data, rowOffset, columns, vector, hasBlockBiases);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5_1NoBias(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            dotRowsQ5_1NoBias(data, rowOffset, rowBytes, columns, vector, 0, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ5_1NoBias(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }
}
