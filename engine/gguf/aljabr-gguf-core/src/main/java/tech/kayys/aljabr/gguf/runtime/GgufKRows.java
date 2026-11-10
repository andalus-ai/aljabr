package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_GROUPS_PER_BLOCK;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufKDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.Q4_DOT_VECTOR_ENABLED;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED;

/**
 * Row walkers for prepared K-quant matrices.
 *
 * <p>Prepared K matrices reuse the same row traversal patterns with different
 * group widths and optional mins. This helper keeps those variants together.</p>
 * @author bhangun
 */
final class GgufKRows {
    private static final int K16_NO_MIN_VECTOR = 1;
    private static final int K16_NO_MIN_SCALAR = 2;
    private static final int K16_PRECOMPUTED_MINS_VECTOR = 3;
    private static final int K16_PRECOMPUTED_MINS_SCALAR = 4;
    private static final int K16_DIRECT_MINS_VECTOR = 5;
    private static final int K16_DIRECT_MINS_SCALAR = 6;
    private static final int K32_NO_MIN_VECTOR = 1;
    private static final int K32_NO_MIN_SCALAR = 2;
    private static final int K32_PRECOMPUTED_MINS_VECTOR = 3;
    private static final int K32_PRECOMPUTED_MINS_SCALAR = 4;
    private static final int K32_DIRECT_MINS_VECTOR = 5;
    private static final int K32_DIRECT_MINS_SCALAR = 6;

    private GgufKRows() {
    }

    static void fillMatVecRowsQ2K(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsQ2K(
                blocksPerRow * QK_K,
                blocksPerRow * QK_GROUPS_PER_BLOCK,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsQ2K(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int rowKernel = vectorGroupSums == null ? k16NoMinKernel() : k16PrecomputedMinsKernel();
        fillMatVecRowsQ2K(
                rowKernel,
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsQ2K(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        if (vectorGroupSums == null) {
            fillMatVecRowsK16PreparedNoMins(
                    rowKernel, rowQuantStride, rowGroupStride, quants, groupScales, vector, output, startRow, endRow);
            return;
        }
        if (rowKernel == K16_PRECOMPUTED_MINS_VECTOR) {
            fillQ2KPrecomputedMinsVector(
                    rowQuantStride,
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    vector,
                    vectorGroupSums,
                    output,
                    startRow,
                    endRow);
            return;
        }
        fillQ2KPrecomputedMinsScalar(
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                output,
                startRow,
                endRow);
    }

    private static void fillQ2KPrecomputedMinsVector(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ2KPreparedVector4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ2KPreparedVector(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector,
                    vectorGroupSums);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    private static void fillQ2KPrecomputedMinsScalar(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ2KPreparedScalar4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ2KPreparedScalar(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector,
                    vectorGroupSums);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    static float dotRowQ2K(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            int row) {
        return dotRowQ2K(
                blocksPerRow * QK_K,
                blocksPerRow * QK_GROUPS_PER_BLOCK,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                row);
    }

    static float dotRowQ2K(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            int row) {
        int rowKernel = vectorGroupSums == null ? k16NoMinKernel() : k16PrecomputedMinsKernel();
        return dotRowQ2K(
                rowKernel,
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                row);
    }

    static float dotRowQ2K(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            int row) {
        int quantsOffset = row * rowQuantStride;
        int matrixGroupBase = row * rowGroupStride;
        if (vectorGroupSums == null) {
            return dotRowK16PreparedNoMins(rowKernel, rowQuantStride, rowGroupStride, quants, groupScales, vector, row);
        }
        return rowKernel == K16_PRECOMPUTED_MINS_VECTOR
                ? dotRowQ2KPreparedVector(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector,
                        vectorGroupSums)
                : dotRowQ2KPreparedScalar(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector,
                        vectorGroupSums);
    }

    static void fillMatVecRowsQ2KDirect(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsQ2KDirect(
                blocksPerRow * QK_K,
                blocksPerRow * QK_GROUPS_PER_BLOCK,
                quants,
                groupScales,
                groupMins,
                vector,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsQ2KDirect(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsQ2KDirect(
                k16DirectMinsKernel(),
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsQ2KDirect(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        if (rowKernel == K16_DIRECT_MINS_VECTOR) {
            fillQ2KDirectMinsVector(
                    rowQuantStride,
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    vector,
                    output,
                    startRow,
                    endRow);
            return;
        }
        fillQ2KDirectMinsScalar(
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                output,
                startRow,
                endRow);
    }

    private static void fillQ2KDirectMinsVector(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ2KPreparedDirectVector4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ2KPreparedDirectVector(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    private static void fillQ2KDirectMinsScalar(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ2KPreparedDirectScalar4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ2KPreparedDirectScalar(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    static float dotRowQ2KDirect(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            int row) {
        return dotRowQ2KDirect(
                blocksPerRow * QK_K,
                blocksPerRow * QK_GROUPS_PER_BLOCK,
                quants,
                groupScales,
                groupMins,
                vector,
                row);
    }

    static float dotRowQ2KDirect(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            int row) {
        return dotRowQ2KDirect(
                k16DirectMinsKernel(), rowQuantStride, rowGroupStride, quants, groupScales, groupMins, vector, row);
    }

    static float dotRowQ2KDirect(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            int row) {
        int quantsOffset = row * rowQuantStride;
        int matrixGroupBase = row * rowGroupStride;
        return rowKernel == K16_DIRECT_MINS_VECTOR
                ? dotRowQ2KPreparedDirectVector(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector)
                : dotRowQ2KPreparedDirectScalar(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector);
    }

    static void fillMatVecRowsK16PreparedNoMins(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsK16PreparedNoMins(
                blocksPerRow * QK_K,
                blocksPerRow * QK_GROUPS_PER_BLOCK,
                quants,
                groupScales,
                vector,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsK16PreparedNoMins(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsK16PreparedNoMins(
                k16NoMinKernel(),
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                vector,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsK16PreparedNoMins(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        if (rowKernel == K16_NO_MIN_VECTOR) {
            fillK16NoMinsVector(
                    rowQuantStride, rowGroupStride, quants, groupScales, vector, output, startRow, endRow);
            return;
        }
        fillK16NoMinsScalar(
                rowQuantStride, rowGroupStride, quants, groupScales, vector, output, startRow, endRow);
    }

    private static void fillK16NoMinsVector(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK16PreparedNoMinsVector4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK16PreparedNoMinsVector(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    private static void fillK16NoMinsScalar(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK16PreparedNoMinsScalar4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK16PreparedNoMinsScalar(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    static float dotRowK16PreparedNoMins(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            int row) {
        return dotRowK16PreparedNoMins(
                blocksPerRow * QK_K,
                blocksPerRow * QK_GROUPS_PER_BLOCK,
                quants,
                groupScales,
                vector,
                row);
    }

    static float dotRowK16PreparedNoMins(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            int row) {
        return dotRowK16PreparedNoMins(
                k16NoMinKernel(), rowQuantStride, rowGroupStride, quants, groupScales, vector, row);
    }

    static float dotRowK16PreparedNoMins(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            int row) {
        int quantsOffset = row * rowQuantStride;
        int matrixGroupBase = row * rowGroupStride;
        return rowKernel == K16_NO_MIN_VECTOR
                ? dotRowK16PreparedNoMinsVector(
                        rowGroupStride, quants, groupScales, quantsOffset, matrixGroupBase, vector)
                : dotRowK16PreparedNoMinsScalar(
                        rowGroupStride, quants, groupScales, quantsOffset, matrixGroupBase, vector);
    }

    private static int k16NoMinKernel() {
        return SIGNED_BYTE_DOT_VECTOR_ENABLED ? K16_NO_MIN_VECTOR : K16_NO_MIN_SCALAR;
    }

    private static int k16PrecomputedMinsKernel() {
        return SIGNED_BYTE_DOT_VECTOR_ENABLED ? K16_PRECOMPUTED_MINS_VECTOR : K16_PRECOMPUTED_MINS_SCALAR;
    }

    private static int k16DirectMinsKernel() {
        return SIGNED_BYTE_DOT_VECTOR_ENABLED ? K16_DIRECT_MINS_VECTOR : K16_DIRECT_MINS_SCALAR;
    }

    static void fillMatVecRowsK32PreparedDirect(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsK32PreparedDirect(
                blocksPerRow * QK_K,
                blocksPerRow * 8,
                quants,
                groupScales,
                groupMins,
                vector,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsK32PreparedDirect(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsK32PreparedDirect(
                k32DirectMinsKernel(),
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsK32PreparedDirect(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        if (rowKernel == K32_DIRECT_MINS_VECTOR) {
            fillK32DirectMinsVector(
                    rowQuantStride,
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    vector,
                    output,
                    startRow,
                    endRow);
            return;
        }
        fillK32DirectMinsScalar(
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                output,
                startRow,
                endRow);
    }

    private static void fillK32DirectMinsVector(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK32GroupsPreparedDirectVector4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK32GroupsPreparedDirectVector(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    private static void fillK32DirectMinsScalar(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK32GroupsPreparedDirectScalar4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK32GroupsPreparedDirectScalar(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    static float dotRowK32PreparedDirect(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            int row) {
        return dotRowK32PreparedDirect(
                blocksPerRow * QK_K,
                blocksPerRow * 8,
                quants,
                groupScales,
                groupMins,
                vector,
                row);
    }

    static float dotRowK32PreparedDirect(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            int row) {
        return dotRowK32PreparedDirect(
                k32DirectMinsKernel(), rowQuantStride, rowGroupStride, quants, groupScales, groupMins, vector, row);
    }

    static float dotRowK32PreparedDirect(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            int row) {
        int quantsOffset = row * rowQuantStride;
        int matrixGroupBase = row * rowGroupStride;
        return rowKernel == K32_DIRECT_MINS_VECTOR
                ? dotRowK32GroupsPreparedDirectVector(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector)
                : dotRowK32GroupsPreparedDirectScalar(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector);
    }

    static void fillMatVecRowsK32Prepared(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        fillMatVecRowsK32Prepared(
                blocksPerRow * QK_K,
                blocksPerRow * 8,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsK32Prepared(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int rowKernel = vectorGroupSums == null ? k32NoMinKernel() : k32PrecomputedMinsKernel();
        fillMatVecRowsK32Prepared(
                rowKernel,
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                output,
                startRow,
                endRow);
    }

    static void fillMatVecRowsK32Prepared(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        if (vectorGroupSums == null) {
            if (rowKernel == K32_NO_MIN_VECTOR) {
                fillK32NoMinsVector(
                        rowQuantStride, rowGroupStride, quants, groupScales, vector, output, startRow, endRow);
                return;
            }
            fillK32NoMinsScalar(
                    rowQuantStride, rowGroupStride, quants, groupScales, vector, output, startRow, endRow);
            return;
        }
        if (rowKernel == K32_PRECOMPUTED_MINS_VECTOR) {
            fillK32PrecomputedMinsVector(
                    rowQuantStride,
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    vector,
                    vectorGroupSums,
                    output,
                    startRow,
                    endRow);
            return;
        }
        fillK32PrecomputedMinsScalar(
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                output,
                startRow,
                endRow);
    }

    private static void fillK32NoMinsVector(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK32GroupsPreparedNoMinsVector4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK32GroupsPreparedNoMinsVector(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    private static void fillK32NoMinsScalar(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK32GroupsPreparedNoMinsScalar4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK32GroupsPreparedNoMinsScalar(
                    rowGroupStride,
                    quants,
                    groupScales,
                    quantsOffset,
                    matrixGroupBase,
                    vector);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    private static void fillK32PrecomputedMinsVector(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK32GroupsPreparedVector4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK32GroupsPreparedVector(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector,
                    vectorGroupSums);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    private static void fillK32PrecomputedMinsScalar(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int quantsOffset = startRow * rowQuantStride;
        int matrixGroupBase = startRow * rowGroupStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourQuantStride = four(rowQuantStride);
        int fourGroupStride = four(rowGroupStride);
        for (; row < tailStart; row += 4) {
            dotRowsK32GroupsPreparedScalar4(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    rowQuantStride,
                    matrixGroupBase,
                    rowGroupStride,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
            quantsOffset += fourQuantStride;
            matrixGroupBase += fourGroupStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowK32GroupsPreparedScalar(
                    rowGroupStride,
                    quants,
                    groupScales,
                    groupMins,
                    quantsOffset,
                    matrixGroupBase,
                    vector,
                    vectorGroupSums);
            quantsOffset += rowQuantStride;
            matrixGroupBase += rowGroupStride;
        }
    }

    static float dotRowK32Prepared(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            int row) {
        return dotRowK32Prepared(
                blocksPerRow * QK_K,
                blocksPerRow * 8,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                row);
    }

    static float dotRowK32Prepared(
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            int row) {
        int rowKernel = vectorGroupSums == null ? k32NoMinKernel() : k32PrecomputedMinsKernel();
        return dotRowK32Prepared(
                rowKernel,
                rowQuantStride,
                rowGroupStride,
                quants,
                groupScales,
                groupMins,
                vector,
                vectorGroupSums,
                row);
    }

    static float dotRowK32Prepared(
            int rowKernel,
            int rowQuantStride,
            int rowGroupStride,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            float[] vector,
            float[] vectorGroupSums,
            int row) {
        int quantsOffset = row * rowQuantStride;
        int matrixGroupBase = row * rowGroupStride;
        if (vectorGroupSums == null) {
            return rowKernel == K32_NO_MIN_VECTOR
                    ? dotRowK32GroupsPreparedNoMinsVector(
                            rowGroupStride, quants, groupScales, quantsOffset, matrixGroupBase, vector)
                    : dotRowK32GroupsPreparedNoMinsScalar(
                            rowGroupStride, quants, groupScales, quantsOffset, matrixGroupBase, vector);
        }
        return rowKernel == K32_PRECOMPUTED_MINS_VECTOR
                ? dotRowK32GroupsPreparedVector(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector,
                        vectorGroupSums)
                : dotRowK32GroupsPreparedScalar(
                        rowGroupStride, quants, groupScales, groupMins, quantsOffset, matrixGroupBase, vector,
                        vectorGroupSums);
    }

    private static int k32NoMinKernel() {
        return Q4_DOT_VECTOR_ENABLED ? K32_NO_MIN_VECTOR : K32_NO_MIN_SCALAR;
    }

    private static int k32PrecomputedMinsKernel() {
        return Q4_DOT_VECTOR_ENABLED ? K32_PRECOMPUTED_MINS_VECTOR : K32_PRECOMPUTED_MINS_SCALAR;
    }

    private static int k32DirectMinsKernel() {
        return Q4_DOT_VECTOR_ENABLED ? K32_DIRECT_MINS_VECTOR : K32_DIRECT_MINS_SCALAR;
    }

    private static int four(int value) {
        int twice = value + value;
        return twice + twice;
    }
}
