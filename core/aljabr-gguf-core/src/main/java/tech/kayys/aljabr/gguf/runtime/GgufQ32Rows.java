package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQ32Dot.*;

import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q32Matrix;

/**
 * Row walkers for prepared Q32-family matrices.
 *
 * <p>This helper keeps prepared row traversal close to the Q32 row-dot reducers
 * while {@link GgufTensorOps} remains the public facade.</p>
 * @author bhangun
 */
final class GgufQ32Rows {
    private GgufQ32Rows() {
    }

    static void fillMatVecRowsQ32(
            Q32Matrix matrix,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int kernel = vectorGroupSums == null ? matrix.noBiasKernel() : matrix.precomputedBiasKernel();
        switch (kernel) {
            case Q32Matrix.ROW_KERNEL_NO_BIAS_VECTOR -> fillNoBiasVector(matrix, vector, output, startRow, endRow);
            case Q32Matrix.ROW_KERNEL_NO_BIAS_SCALAR -> fillNoBiasScalar(matrix, vector, output, startRow, endRow);
            case Q32Matrix.ROW_KERNEL_PRECOMPUTED_BIAS_VECTOR ->
                    fillPrecomputedBiasVector(matrix, vector, vectorGroupSums, output, startRow, endRow);
            default -> fillPrecomputedBiasScalar(matrix, vector, vectorGroupSums, output, startRow, endRow);
        }
    }

    static float dotRowQ32(
            Q32Matrix matrix,
            float[] vector,
            float[] vectorGroupSums,
            int row) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        float[] blockBiases = matrix.blockBiases();
        int matrixBlock = row * blocksPerRow;
        int qBase = row * matrix.quantStride();
        int kernel = vectorGroupSums == null ? matrix.noBiasKernel() : matrix.precomputedBiasKernel();
        return switch (kernel) {
            case Q32Matrix.ROW_KERNEL_NO_BIAS_VECTOR ->
                    dotRowQ32PreparedNoBiasVector(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            case Q32Matrix.ROW_KERNEL_NO_BIAS_SCALAR ->
                    dotRowQ32PreparedNoBiasScalar(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            case Q32Matrix.ROW_KERNEL_PRECOMPUTED_BIAS_VECTOR -> dotRowQ32PreparedVector(
                    blocksPerRow, quants, blockScales, blockBiases, matrixBlock, qBase, vector, vectorGroupSums);
            default -> dotRowQ32PreparedScalar(
                    blocksPerRow, quants, blockScales, blockBiases, matrixBlock, qBase, vector, vectorGroupSums);
        };
    }

    static void fillMatVecRowsQ32Direct(
            Q32Matrix matrix,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        switch (matrix.directBiasKernel()) {
            case Q32Matrix.ROW_KERNEL_DIRECT_BIAS_VECTOR ->
                    fillDirectBiasVector(matrix, vector, output, startRow, endRow);
            default -> fillDirectBiasScalar(matrix, vector, output, startRow, endRow);
        }
    }

    static float dotRowQ32Direct(
            Q32Matrix matrix,
            float[] vector,
            int row) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        float[] blockBiases = matrix.blockBiases();
        int matrixBlock = row * blocksPerRow;
        int qBase = row * matrix.quantStride();
        return switch (matrix.directBiasKernel()) {
            case Q32Matrix.ROW_KERNEL_DIRECT_BIAS_VECTOR -> dotRowQ32PreparedDirectVector(
                    blocksPerRow, quants, blockScales, blockBiases, matrixBlock, qBase, vector);
            default -> dotRowQ32PreparedDirectScalar(
                    blocksPerRow, quants, blockScales, blockBiases, matrixBlock, qBase, vector);
        };
    }

    private static void fillNoBiasVector(
            Q32Matrix matrix,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        int qStride = matrix.quantStride();
        int matrixBlock = startRow * blocksPerRow;
        int qBase = startRow * qStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ32PreparedNoBiasVector4(
                    blocksPerRow,
                    quants,
                    blockScales,
                    matrixBlock,
                    blocksPerRow,
                    qBase,
                    qStride,
                    vector,
                    output,
                    row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ32PreparedNoBiasVector(
                    blocksPerRow,
                    quants,
                    blockScales,
                    matrixBlock,
                    qBase,
                    vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillNoBiasScalar(
            Q32Matrix matrix,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        int qStride = matrix.quantStride();
        int matrixBlock = startRow * blocksPerRow;
        int qBase = startRow * qStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ32PreparedNoBiasScalar4(
                    blocksPerRow,
                    quants,
                    blockScales,
                    matrixBlock,
                    blocksPerRow,
                    qBase,
                    qStride,
                    vector,
                    output,
                    row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ32PreparedNoBiasScalar(
                    blocksPerRow,
                    quants,
                    blockScales,
                    matrixBlock,
                    qBase,
                    vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillPrecomputedBiasVector(
            Q32Matrix matrix,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        float[] blockBiases = matrix.blockBiases();
        int qStride = matrix.quantStride();
        int matrixBlock = startRow * blocksPerRow;
        int qBase = startRow * qStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ32PreparedVector4(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    blocksPerRow,
                    qBase,
                    qStride,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ32PreparedVector(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    qBase,
                    vector,
                    vectorGroupSums);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillPrecomputedBiasScalar(
            Q32Matrix matrix,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        float[] blockBiases = matrix.blockBiases();
        int qStride = matrix.quantStride();
        int matrixBlock = startRow * blocksPerRow;
        int qBase = startRow * qStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ32PreparedScalar4(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    blocksPerRow,
                    qBase,
                    qStride,
                    vector,
                    vectorGroupSums,
                    output,
                    row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ32PreparedScalar(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    qBase,
                    vector,
                    vectorGroupSums);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillDirectBiasVector(
            Q32Matrix matrix,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        float[] blockBiases = matrix.blockBiases();
        int qStride = matrix.quantStride();
        int matrixBlock = startRow * blocksPerRow;
        int qBase = startRow * qStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ32PreparedDirectVector4(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    blocksPerRow,
                    qBase,
                    qStride,
                    vector,
                    output,
                    row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ32PreparedDirectVector(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    qBase,
                    vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillDirectBiasScalar(
            Q32Matrix matrix,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        float[] blockBiases = matrix.blockBiases();
        int qStride = matrix.quantStride();
        int matrixBlock = startRow * blocksPerRow;
        int qBase = startRow * qStride;
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ32PreparedDirectScalar4(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    blocksPerRow,
                    qBase,
                    qStride,
                    vector,
                    output,
                    row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ32PreparedDirectScalar(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    matrixBlock,
                    qBase,
                    vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static int four(int value) {
        int twice = value + value;
        return twice + twice;
    }
}
