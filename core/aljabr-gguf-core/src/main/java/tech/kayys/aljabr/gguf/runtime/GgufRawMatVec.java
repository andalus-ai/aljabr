package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KWorkBuffer;

import java.lang.foreign.MemorySegment;

/**
 * Raw GGUF mat-vec row routing.
 *
 * <p>This helper owns scheduling and per-format raw row selection when a
 * prepared matrix cache is unavailable or intentionally bypassed. Keeping it
 * outside {@link GgufTensorOps} lets the facade focus on API validation and
 * cache selection.</p>
 * @author bhangun
 */
final class GgufRawMatVec {
    private static final int DIRECT_TINY_ROW_LIMIT = 4;
    private static final ThreadLocal<Q4KWorkBuffer> WORK_BUFFER =
            ThreadLocal.withInitial(Q4KWorkBuffer::new);

    private GgufRawMatVec() {
    }

    static void q4K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            q4K(data, columns, rowBytes, vector, output, rowCount, parallel, cachedHasGroupMins.booleanValue());
            return;
        }
        if (rowCount == 1) {
            output[0] = GgufRawDot.q4K(data, columns, vector, cachedHasGroupMins);
            return;
        }
        if (!GgufKMeta.q4RowsHaveMins(data, rowBytes, columns, rowCount)) {
            runQ4KNoMins(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        q4K(data, columns, rowBytes, vector, output, rowCount, parallel, true);
    }

    static void q4K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            boolean hasGroupMins) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q4K(data, columns, vector, hasGroupMins);
            return;
        }
        if (!hasGroupMins) {
            runQ4KNoMins(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        if (shouldUseDirectAffineRows(rowCount)) {
            GgufKRawRows.fillMatVecRowsQ4KDirect(data, columns, rowBytes, vector, output, 0, rowCount, true);
            return;
        }
        float[] vectorGroupSums = GgufSum.q4KVectorGroupSums(vector, columns, WORK_BUFFER.get());
        GgufRawRows.runAux(
                data,
                columns,
                rowBytes,
                vector,
                vectorGroupSums,
                output,
                rowCount,
                parallel,
                GgufKRawRows::fillMatVecRowsQ4K);
    }

    private static void runQ4KNoMins(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        runRawRows(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufKRawRows::fillMatVecRowsQ4KNoMins);
    }

    static void q2K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            q2K(data, columns, rowBytes, vector, output, rowCount, parallel, cachedHasGroupMins.booleanValue());
            return;
        }
        if (rowCount == 1) {
            output[0] = GgufRawDot.q2K(data, columns, vector, cachedHasGroupMins);
            return;
        }
        if (!GgufKMeta.q2RowsHaveMins(data, rowBytes, columns, rowCount)) {
            runQ2KNoMins(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        q2K(data, columns, rowBytes, vector, output, rowCount, parallel, true);
    }

    static void q2K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            boolean hasGroupMins) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q2K(data, columns, vector, hasGroupMins);
            return;
        }
        if (!hasGroupMins) {
            runQ2KNoMins(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        if (shouldUseDirectAffineRows(rowCount)) {
            GgufKRawRows.fillMatVecRowsQ2KDirect(data, columns, rowBytes, vector, output, 0, rowCount, true);
            return;
        }
        float[] vectorGroupSums = GgufSum.vector16GroupSums(vector, columns, WORK_BUFFER.get());
        GgufRawRows.runAux(
                data,
                columns,
                rowBytes,
                vector,
                vectorGroupSums,
                output,
                rowCount,
                parallel,
                GgufKRawRows::fillMatVecRowsQ2K);
    }

    private static void runQ2KNoMins(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        runRawRows(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufKRawRows::fillMatVecRowsQ2KNoMins);
    }

    static void q3K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q3K(data, columns, vector);
            return;
        }
        runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel, GgufKRawRows::fillMatVecRowsQ3K);
    }

    static void q5K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            q5K(data, columns, rowBytes, vector, output, rowCount, parallel, cachedHasGroupMins.booleanValue());
            return;
        }
        if (rowCount == 1) {
            output[0] = GgufRawDot.q5K(data, columns, vector, cachedHasGroupMins);
            return;
        }
        if (!GgufKMeta.q5RowsHaveMins(data, rowBytes, columns, rowCount)) {
            runQ5KNoMins(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        q5K(data, columns, rowBytes, vector, output, rowCount, parallel, true);
    }

    static void q5K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            boolean hasGroupMins) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q5K(data, columns, vector, hasGroupMins);
            return;
        }
        if (!hasGroupMins) {
            runQ5KNoMins(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        if (shouldUseDirectAffineRows(rowCount)) {
            GgufKRawRows.fillMatVecRowsQ5KDirect(data, columns, rowBytes, vector, output, 0, rowCount, true);
            return;
        }
        float[] vectorGroupSums = GgufSum.q4KVectorGroupSums(vector, columns, WORK_BUFFER.get());
        GgufRawRows.runAux(
                data,
                columns,
                rowBytes,
                vector,
                vectorGroupSums,
                output,
                rowCount,
                parallel,
                GgufKRawRows::fillMatVecRowsQ5K);
    }

    private static void runQ5KNoMins(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        runRawRows(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufKRawRows::fillMatVecRowsQ5KNoMins);
    }

    static void q6K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q6K(data, columns, vector);
            return;
        }
        if (shouldUseDirectAffineRows(rowCount)) {
            GgufKRawRows.fillMatVecRowsQ6KDirect(data, columns, rowBytes, vector, output, 0, rowCount);
            return;
        }
        float[] vectorGroupSums = GgufSum.vector16GroupSums(vector, columns, WORK_BUFFER.get());
        GgufRawRows.runAux(
                data,
                columns,
                rowBytes,
                vector,
                vectorGroupSums,
                output,
                rowCount,
                parallel,
                GgufKRawRows::fillMatVecRowsQ6K);
    }

    private static boolean shouldUseDirectAffineRows(int rowCount) {
        return shouldUseDirectRows(rowCount);
    }

    private static boolean shouldUseDirectRows(int rowCount) {
        return rowCount > 1 && rowCount <= DIRECT_TINY_ROW_LIMIT;
    }

    static void q8Raw(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q8(data, route, typeId, columns, vector);
            return;
        }
        switch (route) {
            case GgufQ8Route.Q1_0 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufTqRawRows::fillMatVecRowsQ1_0);
            case GgufQ8Route.TQ1_0 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufTqRawRows::fillMatVecRowsTQ1_0);
            case GgufQ8Route.TQ2_0 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufTqRawRows::fillMatVecRowsTQ2_0);
            case GgufQ8Route.MXFP4 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufNibRawRows::fillMatVecRowsMXFP4);
            case GgufQ8Route.NVFP4 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufNibRawRows::fillMatVecRowsNVFP4);
            case GgufQ8Route.Q8_0 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufQ8RawRows::fillMatVecRowsQ8_0);
            case GgufQ8Route.Q8_1 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufQ8RawRows::fillMatVecRowsQ8_1);
            case GgufQ8Route.Q8_K ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufQ8RawRows::fillMatVecRowsQ8K);
            case GgufQ8Route.IQ4_NL ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufNibRawRows::fillMatVecRowsIQ4NL);
            case GgufQ8Route.IQ4_XS ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufNibRawRows::fillMatVecRowsIQ4XS);
            default -> throw new UnsupportedOperationException("Unsupported GGUF raw Q8-family matvec type id: " + typeId);
        }
    }

    private static void runRawRows(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            GgufRawRows.RawRowFiller filler) {
        if (shouldUseDirectRows(rowCount)) {
            filler.fill(data, columns, rowBytes, vector, output, 0, rowCount);
            return;
        }
        GgufRawRows.run(data, columns, rowBytes, vector, output, rowCount, parallel, filler);
    }

    static void fallback(
            MemorySegment data,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.fallback(data, typeId, columns, vector);
            return;
        }
        runRawTypedRows(
                data,
                typeId,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufRawRows::fillFallback);
    }

    private static void runRawTypedRows(
            MemorySegment data,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            GgufRawRows.RawTypedRowFiller filler) {
        if (shouldUseDirectRows(rowCount)) {
            filler.fill(data, typeId, columns, rowBytes, vector, output, 0, rowCount);
            return;
        }
        GgufRawRows.runTyped(data, typeId, columns, rowBytes, vector, output, rowCount, parallel, filler);
    }

    static void q4_1(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            q4_1(data, columns, rowBytes, vector, output, rowCount, parallel, cachedHasBlockBiases.booleanValue());
            return;
        }
        if (rowCount == 1) {
            output[0] = GgufRawDot.q4_1(data, columns, vector, cachedHasBlockBiases);
            return;
        }
        if (!GgufQ32Meta.q4_1RowsHaveBlockBiases(data, rowBytes, columns, rowCount)) {
            runQ4_1NoBias(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        q4_1(data, columns, rowBytes, vector, output, rowCount, parallel, true);
    }

    static void q4_1(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            boolean hasBlockBiases) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q4_1(data, columns, vector, hasBlockBiases);
            return;
        }
        if (!hasBlockBiases) {
            runQ4_1NoBias(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        if (shouldUseDirectAffineRows(rowCount)) {
            GgufQ32RawRows.fillMatVecRowsQ4_1Direct(data, columns, rowBytes, vector, output, 0, rowCount, true);
            return;
        }
        float[] vectorBlockSums = GgufSum.vector32GroupSums(vector, columns, WORK_BUFFER.get());
        GgufRawRows.runAux(
                data,
                columns,
                rowBytes,
                vector,
                vectorBlockSums,
                output,
                rowCount,
                parallel,
                GgufQ32RawRows::fillMatVecRowsQ4_1);
    }

    private static void runQ4_1NoBias(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        runRawRows(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufQ32RawRows::fillMatVecRowsQ4_1NoBias);
    }

    static void q5_1(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            q5_1(data, columns, rowBytes, vector, output, rowCount, parallel, cachedHasBlockBiases.booleanValue());
            return;
        }
        if (rowCount == 1) {
            output[0] = GgufRawDot.q5_1(data, columns, vector, cachedHasBlockBiases);
            return;
        }
        if (!GgufQ32Meta.q5_1RowsHaveBlockBiases(data, rowBytes, columns, rowCount)) {
            runQ5_1NoBias(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        q5_1(data, columns, rowBytes, vector, output, rowCount, parallel, true);
    }

    static void q5_1(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            boolean hasBlockBiases) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q5_1(data, columns, vector, hasBlockBiases);
            return;
        }
        if (!hasBlockBiases) {
            runQ5_1NoBias(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        if (shouldUseDirectAffineRows(rowCount)) {
            GgufQ32RawRows.fillMatVecRowsQ5_1Direct(data, columns, rowBytes, vector, output, 0, rowCount, true);
            return;
        }
        float[] vectorBlockSums = GgufSum.vector32GroupSums(vector, columns, WORK_BUFFER.get());
        GgufRawRows.runAux(
                data,
                columns,
                rowBytes,
                vector,
                vectorBlockSums,
                output,
                rowCount,
                parallel,
                GgufQ32RawRows::fillMatVecRowsQ5_1);
    }

    private static void runQ5_1NoBias(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        runRawRows(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufQ32RawRows::fillMatVecRowsQ5_1NoBias);
    }

    static void denseRaw(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.dense(data, route, typeId, columns, vector);
            return;
        }
        switch (route) {
            case GgufDenseRoute.F32 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufDenseRows::fillMatVecRowsF32);
            case GgufDenseRoute.F16 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufDenseRows::fillMatVecRowsF16);
            case GgufDenseRoute.BF16 ->
                    runRawRows(data, columns, rowBytes, vector, output, rowCount, parallel,
                            GgufDenseRows::fillMatVecRowsBF16);
            default -> throw new UnsupportedOperationException("Unsupported GGUF dense raw matvec type id: " + typeId);
        }
    }

    static void q32Raw(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            q32Raw(data, route, typeId, columns, rowBytes, vector, output, rowCount, parallel,
                    cachedHasBlockBiases.booleanValue());
            return;
        }
        if (rowCount == 1) {
            output[0] = GgufRawDot.q32(data, route, typeId, columns, vector, cachedHasBlockBiases);
            return;
        }
        switch (route) {
            case GgufQ32Route.Q4_1 -> q4_1(data, columns, rowBytes, vector, output, rowCount, parallel, cachedHasBlockBiases);
            case GgufQ32Route.Q5_1 -> q5_1(data, columns, rowBytes, vector, output, rowCount, parallel, cachedHasBlockBiases);
            case GgufQ32Route.Q4_0 -> runQ4_0(data, columns, rowBytes, vector, output, rowCount, parallel);
            case GgufQ32Route.Q5_0 -> runQ5_0(data, columns, rowBytes, vector, output, rowCount, parallel);
            default -> throw new UnsupportedOperationException("Unsupported GGUF raw Q32-family matvec type id: " + typeId);
        }
    }

    static void q32Raw(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            boolean hasBlockBiases) {
        if (rowCount == 1) {
            output[0] = GgufRawDot.q32(data, route, typeId, columns, vector, hasBlockBiases);
            return;
        }
        switch (route) {
            case GgufQ32Route.Q4_1 -> q4_1(data, columns, rowBytes, vector, output, rowCount, parallel, hasBlockBiases);
            case GgufQ32Route.Q5_1 -> q5_1(data, columns, rowBytes, vector, output, rowCount, parallel, hasBlockBiases);
            case GgufQ32Route.Q4_0 -> runQ4_0(data, columns, rowBytes, vector, output, rowCount, parallel);
            case GgufQ32Route.Q5_0 -> runQ5_0(data, columns, rowBytes, vector, output, rowCount, parallel);
            default -> throw new UnsupportedOperationException("Unsupported GGUF raw Q32-family matvec type id: " + typeId);
        }
    }

    private static void runQ4_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        runRawRows(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufQ32RawRows::fillMatVecRowsQ4_0);
    }

    private static void runQ5_0(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        runRawRows(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                GgufQ32RawRows::fillMatVecRowsQ5_0);
    }

}
