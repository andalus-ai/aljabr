package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQ8RawDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.RAW_Q8_VECTOR_DOT_ENABLED;

import java.lang.foreign.MemorySegment;

/**
 * Row walkers for raw Q8 GGUF matrices.
 *
 * <p>Q8_0, Q8_1, and Q8_K share vector/scalar row traversal over raw
 * {@link MemorySegment} data while delegating arithmetic to {@link GgufQ8RawDot}.</p>
 * @author bhangun
 */
final class GgufQ8RawRows {
    private GgufQ8RawRows() {
    }

    static void fillMatVecRowsQ8_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        if (RAW_Q8_VECTOR_DOT_ENABLED) {
            int row = startRow;
            int unrolledEnd = endRow - 4;
            for (; row <= unrolledEnd; row += 4) {
                dotRowsQ8_0Vector4(data, rowOffset, rowBytes, columns, vector, output, row);
                rowOffset += 4L * rowBytes;
            }
            for (; row < endRow; row++) {
                output[row] = dotRowQ8_0Vector(data, rowOffset, columns, vector, 0);
                rowOffset += rowBytes;
            }
            return;
        }
        fillMatVecRowsQ8_0Scalar(data, columns, rowBytes, vector, output, startRow, endRow);
    }

    static void fillMatVecRowsQ8_0Scalar(
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
            dotRowsQ8_0Scalar4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8_0Scalar(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ8_1(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        if (RAW_Q8_VECTOR_DOT_ENABLED) {
            int row = startRow;
            int unrolledEnd = endRow - 4;
            for (; row <= unrolledEnd; row += 4) {
                dotRowsQ8_1Vector4(data, rowOffset, rowBytes, columns, vector, output, row);
                rowOffset += 4L * rowBytes;
            }
            for (; row < endRow; row++) {
                output[row] = dotRowQ8_1Vector(data, rowOffset, columns, vector, 0);
                rowOffset += rowBytes;
            }
            return;
        }
        fillMatVecRowsQ8_1Scalar(data, columns, rowBytes, vector, output, startRow, endRow);
    }

    static void fillMatVecRowsQ8_1Scalar(
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
            dotRowsQ8_1Scalar4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8_1Scalar(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ8K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        if (RAW_Q8_VECTOR_DOT_ENABLED) {
            int row = startRow;
            int unrolledEnd = endRow - 4;
            for (; row <= unrolledEnd; row += 4) {
                dotRowsQ8KVector4(data, rowOffset, rowBytes, columns, vector, output, row);
                rowOffset += 4L * rowBytes;
            }
            for (; row < endRow; row++) {
                output[row] = dotRowQ8KVector(data, rowOffset, columns, vector, 0);
                rowOffset += rowBytes;
            }
            return;
        }
        fillMatVecRowsQ8KScalar(data, columns, rowBytes, vector, output, startRow, endRow);
    }

    static void fillMatVecRowsQ8KScalar(
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
            dotRowsQ8KScalar4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8KScalar(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }
}
