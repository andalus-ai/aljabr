package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q2KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q32Matrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q3KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KWorkBuffer;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q5KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q6KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q8Matrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

import java.util.Objects;

/**
 * Prepared GGUF mat-vec row scheduling.
 *
 * <p>Prepared matrices have already unpacked quant data into cache-friendly
 * byte arrays. This helper keeps their row execution separate from the public
 * tensor facade and from raw GGUF row decoding.</p>
 * @author bhangun
 */
final class GgufPrepRows {
    private static final ThreadLocal<Q4KWorkBuffer> WORK_BUFFER =
            ThreadLocal.withInitial(Q4KWorkBuffer::new);

    private GgufPrepRows() {
    }

    static void q4K(Q4KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        q4K(matrix, vector, output, rowCount, parallel, null);
    }

    static void rows(
            PreparedMatrix matrix,
            GgufPreparedCachePolicy.Family family,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(family, "family");
        switch (family) {
            case Q32 -> q32((Q32Matrix) matrix, vector, output, rowCount, parallel);
            case Q2K -> q2K((Q2KMatrix) matrix, vector, output, rowCount, parallel);
            case Q3K -> q3K((Q3KMatrix) matrix, vector, output, rowCount, parallel);
            case Q4K -> q4K((Q4KMatrix) matrix, vector, output, rowCount, parallel);
            case Q5K -> q5K((Q5KMatrix) matrix, vector, output, rowCount, parallel);
            case Q6K -> q6K((Q6KMatrix) matrix, vector, output, rowCount, parallel);
            case Q8 -> q8((Q8Matrix) matrix, vector, output, rowCount, parallel);
        }
    }

    static void rowsTrusted(
            PreparedMatrix matrix,
            GgufPreparedCachePolicy.Family family,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        switch (family) {
            case Q32 -> q32Trusted((Q32Matrix) matrix, columns, vector, output, rowCount, parallel);
            case Q2K -> q2KTrusted((Q2KMatrix) matrix, columns, vector, output, rowCount, parallel);
            case Q3K -> q3KTrusted((Q3KMatrix) matrix, columns, vector, output, rowCount, parallel);
            case Q4K -> q4KTrusted((Q4KMatrix) matrix, columns, vector, output, rowCount, parallel, null);
            case Q5K -> q5KTrusted((Q5KMatrix) matrix, columns, vector, output, rowCount, parallel);
            case Q6K -> q6KTrusted((Q6KMatrix) matrix, columns, vector, output, rowCount, parallel);
            case Q8 -> q8Trusted((Q8Matrix) matrix, columns, vector, output, rowCount, parallel);
        }
    }

    static void q2K(Q2KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        int columns = GgufRows.checkPreparedMatVec(matrix.columns(), matrix.rows(), vector, output, rowCount);
        if (rowCount == 0) {
            return;
        }
        q2KTrusted(matrix, columns, vector, output, rowCount, parallel);
    }

    private static void q2KTrusted(
            Q2KMatrix matrix,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        int quantStride = matrix.quantStride();
        int groupStride = matrix.groupStride();
        int noMinKernel = matrix.noMinKernel();
        int precomputedMinsKernel = matrix.precomputedMinsKernel();
        int directMinsKernel = matrix.directMinsKernel();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        float[] groupMins = matrix.groupMins();
        boolean hasGroupMins = matrix.hasGroupMins();
        if (hasGroupMins && rowCount == 1) {
            output[0] = GgufKRows.dotRowQ2KDirect(
                    directMinsKernel, quantStride, groupStride, quants, groupScales, groupMins, vector, 0);
            return;
        }
        if (hasGroupMins && shouldUseDirectAffineRows(rowCount)) {
            GgufKRows.fillMatVecRowsQ2KDirect(
                    directMinsKernel,
                    quantStride,
                    groupStride,
                    quants,
                    groupScales,
                    groupMins,
                    vector,
                    output,
                    0,
                    rowCount);
            return;
        }
        float[] vectorGroupSums = hasGroupMins
                ? GgufSum.vector16GroupSums(vector, columns, WORK_BUFFER.get())
                : null;
        int rowKernel = hasGroupMins ? precomputedMinsKernel : noMinKernel;
        if (rowCount == 1) {
            output[0] = GgufKRows.dotRowQ2K(
                    rowKernel, quantStride, groupStride, quants, groupScales, groupMins, vector, vectorGroupSums, 0);
            return;
        }
        GgufRows.runPreparedRows(parallel, rowCount, columns, (start, end) ->
                GgufKRows.fillMatVecRowsQ2K(
                        rowKernel,
                        quantStride,
                        groupStride,
                        quants,
                        groupScales,
                        groupMins,
                        vector,
                        vectorGroupSums,
                        output,
                        start,
                        end));
    }

    static void q3K(Q3KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        int columns = GgufRows.checkPreparedMatVec(matrix.columns(), matrix.rows(), vector, output, rowCount);
        if (rowCount == 0) {
            return;
        }
        q3KTrusted(matrix, columns, vector, output, rowCount, parallel);
    }

    private static void q3KTrusted(
            Q3KMatrix matrix,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        int quantStride = matrix.quantStride();
        int groupStride = matrix.groupStride();
        int rowKernel = matrix.rowKernel();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        if (rowCount == 1) {
            output[0] = GgufKRows.dotRowK16PreparedNoMins(
                    rowKernel, quantStride, groupStride, quants, groupScales, vector, 0);
            return;
        }
        GgufRows.runPreparedRows(parallel, rowCount, columns, (start, end) ->
                GgufKRows.fillMatVecRowsK16PreparedNoMins(
                        rowKernel,
                        quantStride,
                        groupStride,
                        quants,
                        groupScales,
                        vector,
                        output,
                        start,
                        end));
    }

    static void q32(Q32Matrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        int columns = GgufRows.checkPreparedMatVec(matrix.columns(), matrix.rows(), vector, output, rowCount);
        if (rowCount == 0) {
            return;
        }
        q32Trusted(matrix, columns, vector, output, rowCount, parallel);
    }

    private static void q32Trusted(
            Q32Matrix matrix,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        boolean hasBlockBiases = matrix.hasBlockBiases();
        if (hasBlockBiases && rowCount == 1) {
            output[0] = GgufQ32Rows.dotRowQ32Direct(matrix, vector, 0);
            return;
        }
        if (hasBlockBiases && shouldUseDirectAffineRows(rowCount)) {
            GgufQ32Rows.fillMatVecRowsQ32Direct(matrix, vector, output, 0, rowCount);
            return;
        }
        float[] vectorGroupSums = hasBlockBiases
                ? GgufSum.vector32GroupSums(vector, columns, WORK_BUFFER.get())
                : null;
        if (rowCount == 1) {
            output[0] = GgufQ32Rows.dotRowQ32(matrix, vector, vectorGroupSums, 0);
            return;
        }
        GgufRows.runPreparedRows(parallel, rowCount, columns, (start, end) ->
                GgufQ32Rows.fillMatVecRowsQ32(
                        matrix,
                        vector,
                        vectorGroupSums,
                        output,
                        start,
                        end));
    }

    static void q5K(Q5KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        int columns = GgufRows.checkPreparedMatVec(matrix.columns(), matrix.rows(), vector, output, rowCount);
        if (rowCount == 0) {
            return;
        }
        q5KTrusted(matrix, columns, vector, output, rowCount, parallel);
    }

    private static void q5KTrusted(
            Q5KMatrix matrix,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        int quantStride = matrix.quantStride();
        int groupStride = matrix.groupStride();
        int noMinKernel = matrix.noMinKernel();
        int precomputedMinsKernel = matrix.precomputedMinsKernel();
        int directMinsKernel = matrix.directMinsKernel();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        float[] groupMins = matrix.groupMins();
        boolean hasGroupMins = matrix.hasGroupMins();
        if (hasGroupMins && rowCount == 1) {
            output[0] = GgufKRows.dotRowK32PreparedDirect(
                    directMinsKernel, quantStride, groupStride, quants, groupScales, groupMins, vector, 0);
            return;
        }
        if (hasGroupMins && shouldUseDirectAffineRows(rowCount)) {
            GgufKRows.fillMatVecRowsK32PreparedDirect(
                    directMinsKernel,
                    quantStride,
                    groupStride,
                    quants,
                    groupScales,
                    groupMins,
                    vector,
                    output,
                    0,
                    rowCount);
            return;
        }
        float[] vectorGroupSums = hasGroupMins
                ? GgufSum.q4KVectorGroupSums(vector, columns, WORK_BUFFER.get())
                : null;
        int rowKernel = hasGroupMins ? precomputedMinsKernel : noMinKernel;
        if (rowCount == 1) {
            output[0] = GgufKRows.dotRowK32Prepared(
                    rowKernel, quantStride, groupStride, quants, groupScales, groupMins, vector, vectorGroupSums, 0);
            return;
        }
        GgufRows.runPreparedRows(parallel, rowCount, columns, (start, end) ->
                GgufKRows.fillMatVecRowsK32Prepared(
                        rowKernel,
                        quantStride,
                        groupStride,
                        quants,
                        groupScales,
                        groupMins,
                        vector,
                        vectorGroupSums,
                        output,
                        start,
                        end));
    }

    static void q6K(Q6KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        int columns = GgufRows.checkPreparedMatVec(matrix.columns(), matrix.rows(), vector, output, rowCount);
        if (rowCount == 0) {
            return;
        }
        q6KTrusted(matrix, columns, vector, output, rowCount, parallel);
    }

    private static void q6KTrusted(
            Q6KMatrix matrix,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        int quantStride = matrix.quantStride();
        int groupStride = matrix.groupStride();
        int rowKernel = matrix.rowKernel();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        if (rowCount == 1) {
            output[0] = GgufKRows.dotRowK16PreparedNoMins(
                    rowKernel, quantStride, groupStride, quants, groupScales, vector, 0);
            return;
        }
        GgufRows.runPreparedRows(parallel, rowCount, columns, (start, end) ->
                GgufKRows.fillMatVecRowsK16PreparedNoMins(
                        rowKernel,
                        quantStride,
                        groupStride,
                        quants,
                        groupScales,
                        vector,
                        output,
                        start,
                        end));
    }

    static void q8(Q8Matrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        int columns = GgufRows.checkPreparedMatVec(matrix.columns(), matrix.rows(), vector, output, rowCount);
        if (rowCount == 0) {
            return;
        }
        q8Trusted(matrix, columns, vector, output, rowCount, parallel);
    }

    private static void q8Trusted(
            Q8Matrix matrix,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (rowCount == 1) {
            output[0] = GgufQ8Rows.dotRowQ8(matrix, vector, 0);
            return;
        }
        GgufRows.runPreparedRows(parallel, rowCount, columns, (start, end) ->
                GgufQ8Rows.fillMatVecRowsQ8(
                        matrix,
                        vector,
                        output,
                        start,
                        end));
    }

    static void q4K(
            Q4KMatrix matrix,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Q4KWorkBuffer workBuffer) {
        Objects.requireNonNull(matrix, "matrix");
        int columns = GgufRows.checkPreparedMatVec(matrix.columns(), matrix.rows(), vector, output, rowCount);
        if (rowCount == 0) {
            return;
        }
        q4KTrusted(matrix, columns, vector, output, rowCount, parallel, workBuffer);
    }

    private static void q4KTrusted(
            Q4KMatrix matrix,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Q4KWorkBuffer workBuffer) {
        int quantStride = matrix.quantStride();
        int groupStride = matrix.groupStride();
        int noMinKernel = matrix.noMinKernel();
        int precomputedMinsKernel = matrix.precomputedMinsKernel();
        int directMinsKernel = matrix.directMinsKernel();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        float[] groupMins = matrix.groupMins();
        boolean hasGroupMins = matrix.hasGroupMins();
        if (hasGroupMins && rowCount == 1) {
            output[0] = GgufKRows.dotRowK32PreparedDirect(
                    directMinsKernel, quantStride, groupStride, quants, groupScales, groupMins, vector, 0);
            return;
        }
        if (hasGroupMins && workBuffer == null && shouldUseDirectAffineRows(rowCount)) {
            GgufKRows.fillMatVecRowsK32PreparedDirect(
                    directMinsKernel,
                    quantStride,
                    groupStride,
                    quants,
                    groupScales,
                    groupMins,
                    vector,
                    output,
                    0,
                    rowCount);
            return;
        }
        float[] vectorGroupSums = hasGroupMins
                ? GgufSum.q4KVectorGroupSums(vector, columns, workBuffer(workBuffer))
                : null;
        int rowKernel = hasGroupMins ? precomputedMinsKernel : noMinKernel;
        if (rowCount == 1) {
            output[0] = GgufKRows.dotRowK32Prepared(
                    rowKernel, quantStride, groupStride, quants, groupScales, groupMins, vector, vectorGroupSums, 0);
            return;
        }
        GgufRows.runPreparedRows(parallel, rowCount, columns, (start, end) ->
                GgufKRows.fillMatVecRowsK32Prepared(
                        rowKernel,
                        quantStride,
                        groupStride,
                        quants,
                        groupScales,
                        groupMins,
                        vector,
                        vectorGroupSums,
                        output,
                        start,
                        end));
    }

    private static Q4KWorkBuffer workBuffer(Q4KWorkBuffer workBuffer) {
        return workBuffer == null ? WORK_BUFFER.get() : workBuffer;
    }

    private static boolean shouldUseDirectAffineRows(int rowCount) {
        return GgufRows.shouldRunPreparedRowsDirect(rowCount);
    }
}
