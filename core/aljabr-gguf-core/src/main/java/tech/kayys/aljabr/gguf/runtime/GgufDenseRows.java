package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufDenseDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.DENSE_BF16_VECTOR_DOT_ENABLED;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.DENSE_F32_VECTOR_DOT_ENABLED;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_VECTOR_LANES;

import java.lang.foreign.MemorySegment;

/**
 * Row walkers for dense raw GGUF matrices.
 *
 * <p>Dense matrices share simple row iteration and delegate all arithmetic to
 * {@link GgufDenseDot}. Keeping the walkers here avoids duplicating dispatch
 * decisions in the public tensor facade.</p>
 * @author bhangun
 */
final class GgufDenseRows {
    private GgufDenseRows() {
    }

    static void fillMatVecRowsF32(
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
        if (DENSE_F32_VECTOR_DOT_ENABLED && columns >= FLOAT_SUM_VECTOR_LANES) {
            for (; row < tailStart; row += 4) {
                dotRowsF32Vector4(data, rowOffset, rowBytes, columns, vector, output, row);
                rowOffset += fourRowBytes;
            }
            for (; row < endRow; row++) {
                output[row] = dotRowF32Vector(data, rowOffset, columns, vector, 0);
                rowOffset += rowBytes;
            }
            return;
        }
        fillMatVecRowsF32Scalar(data, columns, rowBytes, vector, output, startRow, endRow);
    }

    static void fillMatVecRowsF32Scalar(
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
            dotRowsF32Scalar4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowF32Scalar(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsF16(
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
        long fourRowBytes = 4L * rowBytes;
        for (; row < tailStart; row += 4) {
            dotRowsF16Scalar4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowF16(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsBF16(
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
        if (DENSE_BF16_VECTOR_DOT_ENABLED && columns >= FLOAT_SUM_VECTOR_LANES) {
            for (; row < tailStart; row += 4) {
                dotRowsBF16Vector4(data, rowOffset, rowBytes, columns, vector, output, row);
                rowOffset += fourRowBytes;
            }
            for (; row < endRow; row++) {
                output[row] = dotRowBF16Vector(data, rowOffset, columns, vector, 0);
                rowOffset += rowBytes;
            }
            return;
        }
        fillMatVecRowsBF16Scalar(data, columns, rowBytes, vector, output, startRow, endRow);
    }

    static void fillMatVecRowsBF16Scalar(
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
            dotRowsBF16Scalar4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowBF16Scalar(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }
}
