package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;

/**
 * Shared raw GGUF row execution helpers.
 *
 * <p>Raw mat-vec kernels all need the same sequential, parallel, and confined
 * {@link MemorySegment} fallback policy. Keeping that policy here prevents the
 * per-format router from carrying three copies of the same scheduling code.</p>
 * @author bhangun
 */
final class GgufRawRows {
    private GgufRawRows() {
    }

    static void run(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            RawRowFiller filler) {
        if (!parallel || rowCount <= 1) {
            filler.fill(data, columns, rowBytes, vector, output, 0, rowCount);
            return;
        }
        int chunks = GgufRows.rawRowChunks(data, parallel, rowCount, columns);
        if (chunks > 0) {
            try {
                GgufRows.runRawParallelRows(rowCount, chunks, (start, end) ->
                        filler.fill(data, columns, rowBytes, vector, output, start, end));
                return;
            } catch (WrongThreadException ignored) {
                // Confined FFM segments cannot be read by worker threads; keep correctness and fall back.
            }
        }
        filler.fill(data, columns, rowBytes, vector, output, 0, rowCount);
    }

    static void runAux(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] aux,
            float[] output,
            int rowCount,
            boolean parallel,
            RawAuxRowFiller filler) {
        if (!parallel || rowCount <= 1) {
            filler.fill(data, columns, rowBytes, vector, aux, output, 0, rowCount);
            return;
        }
        int chunks = GgufRows.rawRowChunks(data, parallel, rowCount, columns);
        if (chunks > 0) {
            try {
                GgufRows.runRawParallelRows(rowCount, chunks, (start, end) ->
                        filler.fill(data, columns, rowBytes, vector, aux, output, start, end));
                return;
            } catch (WrongThreadException ignored) {
                // Confined FFM segments cannot be read by worker threads; keep correctness and fall back.
            }
        }
        filler.fill(data, columns, rowBytes, vector, aux, output, 0, rowCount);
    }

    static void runTyped(
            MemorySegment data,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            RawTypedRowFiller filler) {
        if (!parallel || rowCount <= 1) {
            filler.fill(data, typeId, columns, rowBytes, vector, output, 0, rowCount);
            return;
        }
        int chunks = GgufRows.rawRowChunks(data, parallel, rowCount, columns);
        if (chunks > 0) {
            try {
                GgufRows.runRawParallelRows(rowCount, chunks, (start, end) ->
                        filler.fill(data, typeId, columns, rowBytes, vector, output, start, end));
                return;
            } catch (WrongThreadException ignored) {
                // Confined FFM segments cannot be read by worker threads; keep correctness and fall back.
            }
        }
        filler.fill(data, typeId, columns, rowBytes, vector, output, 0, rowCount);
    }

    static void fillFallback(
            MemorySegment data,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        long rowOffset = startRow * rowBytes;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        long twoRowBytes = rowBytes + rowBytes;
        long threeRowBytes = twoRowBytes + rowBytes;
        long fourRowBytes = threeRowBytes + rowBytes;
        for (; row < tailStart; row += 4) {
            output[row] = GgufRowDot.row(data, rowOffset, typeId, columns, vector, 0);
            output[row + 1] = GgufRowDot.row(data, rowOffset + rowBytes, typeId, columns, vector, 0);
            output[row + 2] = GgufRowDot.row(data, rowOffset + twoRowBytes, typeId, columns, vector, 0);
            output[row + 3] = GgufRowDot.row(data, rowOffset + threeRowBytes, typeId, columns, vector, 0);
            rowOffset += fourRowBytes;
        }
        for (; row < endRow; row++) {
            output[row] = GgufRowDot.row(data, rowOffset, typeId, columns, vector, 0);
            rowOffset += rowBytes;
        }
    }

    @FunctionalInterface
    interface RawRowFiller {
        void fill(
                MemorySegment data,
                int columns,
                long rowBytes,
                float[] vector,
                float[] output,
                int startRow,
                int endRow);
    }

    @FunctionalInterface
    interface RawAuxRowFiller {
        void fill(
                MemorySegment data,
                int columns,
                long rowBytes,
                float[] vector,
                float[] aux,
                float[] output,
                int startRow,
                int endRow);
    }

    @FunctionalInterface
    interface RawTypedRowFiller {
        void fill(
                MemorySegment data,
                int typeId,
                int columns,
                long rowBytes,
                float[] vector,
                float[] output,
                int startRow,
                int endRow);
    }
}
