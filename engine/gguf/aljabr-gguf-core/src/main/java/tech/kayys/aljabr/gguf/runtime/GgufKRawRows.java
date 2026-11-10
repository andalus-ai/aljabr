package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;

/**
 * Row walkers for raw K-quant mat-vec paths.
 *
 * <p>Raw K formats share the same row-offset traversal even though each dot
 * kernel has different packing and min-correction rules. Keeping the walkers
 * here leaves {@link GgufTensorOps} to choose the path instead of owning every
 * per-format loop.</p>
 * @author bhangun
 */
final class GgufKRawRows {
    private static final ThreadLocal<int[]> Q3K_SCALES = ThreadLocal.withInitial(() -> new int[16]);

    private GgufKRawRows() {
    }

    static void fillMatVecRowsQ2K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            GgufQ2RawDot.dotRowsQ2KWithGroupSums4(
                    data, rowOffset, rowBytes, columns, vector, vectorGroupSums, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ2RawDot.dotRowQ2KWithGroupSums(data, rowOffset, columns, vector, vectorGroupSums);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ2KDirect(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            fillMatVecRowsQ2KDirect(data, columns, rowBytes, vector, output, startRow, endRow,
                    cachedHasGroupMins.booleanValue());
            return;
        }
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q2K(data, rowOffset, columns, vector, cachedHasGroupMins);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ2KDirect(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            boolean hasGroupMins) {
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q2K(data, rowOffset, columns, vector, hasGroupMins);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ2KNoMins(
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
            GgufQ2RawDot.dotRowsQ2KNoMins4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ2RawDot.dotRowQ2KNoMins(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ3K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int[] scales = Q3K_SCALES.get();
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            GgufQ3RawDot.dotRowsQ3K4(data, rowOffset, rowBytes, columns, vector, output, row, scales);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ3RawDot.dotRowQ3K(data, rowOffset, columns, vector, 0, scales);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            GgufQ4RawDot.dotRowsQ4KWithGroupSums4(
                    data, rowOffset, rowBytes, columns, vector, vectorGroupSums, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ4RawDot.dotRowQ4KWithGroupSums(data, rowOffset, columns, vector, vectorGroupSums);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4KDirect(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            fillMatVecRowsQ4KDirect(data, columns, rowBytes, vector, output, startRow, endRow,
                    cachedHasGroupMins.booleanValue());
            return;
        }
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q4K(data, rowOffset, columns, vector, cachedHasGroupMins);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4KDirect(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            boolean hasGroupMins) {
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q4K(data, rowOffset, columns, vector, hasGroupMins);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ4KNoMins(
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
            GgufQ4RawDot.dotRowsQ4KNoMins4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ4RawDot.dotRowQ4KNoMins(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            GgufQ5RawDot.dotRowsQ5KWithGroupSums4(
                    data, rowOffset, rowBytes, columns, vector, vectorGroupSums, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ5RawDot.dotRowQ5KWithGroupSums(data, rowOffset, columns, vector, vectorGroupSums);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5KDirect(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            fillMatVecRowsQ5KDirect(data, columns, rowBytes, vector, output, startRow, endRow,
                    cachedHasGroupMins.booleanValue());
            return;
        }
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q5K(data, rowOffset, columns, vector, cachedHasGroupMins);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5KDirect(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            boolean hasGroupMins) {
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufRawDot.q5K(data, rowOffset, columns, vector, hasGroupMins);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ5KNoMins(
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
            GgufQ5RawDot.dotRowsQ5KNoMins4(data, rowOffset, rowBytes, columns, vector, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ5RawDot.dotRowQ5KNoMins(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ6KDirect(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        for (int row = startRow; row < endRow; row++) {
            output[row] = GgufQ6RawDot.dotRowQ6K(data, rowOffset, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    static void fillMatVecRowsQ6K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int unrolledEnd = endRow - 4;
        for (; row <= unrolledEnd; row += 4) {
            GgufQ6RawDot.dotRowsQ6KWithGroupSums4(
                    data, rowOffset, rowBytes, columns, vector, vectorGroupSums, output, row);
            rowOffset += 4L * rowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufQ6RawDot.dotRowQ6KWithGroupSums(data, rowOffset, columns, vector, vectorGroupSums);
            rowOffset += rowBytes;
        }
    }
}
