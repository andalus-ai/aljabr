package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQ8Dot.*;

import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q8Matrix;

/**
 * Row walkers for prepared Q8-family matrices.
 *
 * <p>Prepared Q8 matrices can use several block widths. This helper chooses the
 * matching row-dot reducer and keeps that dispatch out of the public facade.</p>
 * @author bhangun
 */
final class GgufQ8Rows {
    private GgufQ8Rows() {
    }

    static void fillMatVecRowsQ8(
            Q8Matrix matrix,
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
        switch (matrix.rowKernel()) {
            case Q8Matrix.ROW_KERNEL_32_VECTOR -> fill32RowsVector(
                    blocksPerRow,
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
            case Q8Matrix.ROW_KERNEL_32_SCALAR -> fill32RowsScalar(
                    blocksPerRow,
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
            case Q8Matrix.ROW_KERNEL_16_VECTOR -> fill16RowsVector(
                    blocksPerRow,
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
            case Q8Matrix.ROW_KERNEL_16_SCALAR -> fill16RowsScalar(
                    blocksPerRow,
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
            case Q8Matrix.ROW_KERNEL_WIDE_VECTOR -> fillWideRowsVector(
                    blocksPerRow,
                    matrix.blockSize(),
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
            case Q8Matrix.ROW_KERNEL_WIDE_SCALAR -> fillWideRowsScalar(
                    blocksPerRow,
                    matrix.blockSize(),
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
            case Q8Matrix.ROW_KERNEL_BLOCK_VECTOR -> fillBlockRowsVector(
                    blocksPerRow,
                    matrix.blockSize(),
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
            default -> fillBlockRowsScalar(
                    blocksPerRow,
                    matrix.blockSize(),
                    quants,
                    blockScales,
                    vector,
                    output,
                    startRow,
                    endRow,
                    matrixBlock,
                    qBase,
                    qStride);
        }
    }

    static float dotRowQ8(
            Q8Matrix matrix,
            float[] vector,
            int row) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        int matrixBlock = row * blocksPerRow;
        int qBase = row * matrix.quantStride();
        return switch (matrix.rowKernel()) {
            case Q8Matrix.ROW_KERNEL_32_VECTOR ->
                    dotRowQ8Prepared32Vector(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            case Q8Matrix.ROW_KERNEL_32_SCALAR ->
                    dotRowQ8Prepared32Scalar(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            case Q8Matrix.ROW_KERNEL_16_VECTOR ->
                    dotRowQ8Prepared16Vector(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            case Q8Matrix.ROW_KERNEL_16_SCALAR ->
                    dotRowQ8Prepared16Scalar(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            case Q8Matrix.ROW_KERNEL_WIDE_VECTOR -> dotRowQ8PreparedWideVector(
                    blocksPerRow, matrix.blockSize(), quants, blockScales, matrixBlock, qBase, vector);
            case Q8Matrix.ROW_KERNEL_WIDE_SCALAR -> dotRowQ8PreparedWideScalar(
                    blocksPerRow, matrix.blockSize(), quants, blockScales, matrixBlock, qBase, vector);
            case Q8Matrix.ROW_KERNEL_BLOCK_VECTOR -> dotRowQ8PreparedBlockVector(
                    blocksPerRow, matrix.blockSize(), quants, blockScales, matrixBlock, qBase, vector);
            default -> dotRowQ8PreparedBlockScalar(
                    blocksPerRow, matrix.blockSize(), quants, blockScales, matrixBlock, qBase, vector);
        };
    }

    private static void fill32RowsVector(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8Prepared32Vector4(
                    blocksPerRow, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8Prepared32Vector(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fill32RowsScalar(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8Prepared32Scalar4(
                    blocksPerRow, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8Prepared32Scalar(
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

    private static void fill16RowsVector(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8Prepared16Vector4(
                    blocksPerRow, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8Prepared16Vector(blocksPerRow, quants, blockScales, matrixBlock, qBase, vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fill16RowsScalar(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8Prepared16Scalar4(
                    blocksPerRow, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8Prepared16Scalar(
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

    private static void fillWideRowsVector(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8PreparedWideVector4(
                    blocksPerRow, blockSize, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8PreparedWideVector(
                    blocksPerRow, blockSize, quants, blockScales, matrixBlock, qBase, vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillWideRowsScalar(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8PreparedWideScalar4(
                    blocksPerRow, blockSize, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8PreparedWideScalar(
                    blocksPerRow,
                    blockSize,
                    quants,
                    blockScales,
                    matrixBlock,
                    qBase,
                    vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillBlockRowsVector(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8PreparedBlockVector4(
                    blocksPerRow, blockSize, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8PreparedBlockVector(
                    blocksPerRow, blockSize, quants, blockScales, matrixBlock, qBase, vector);
            matrixBlock += blocksPerRow;
            qBase += qStride;
        }
    }

    private static void fillBlockRowsScalar(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            float[] vector,
            float[] output,
            int startRow,
            int endRow,
            int matrixBlock,
            int qBase,
            int qStride) {
        int row = startRow;
        int tailStart = endRow - ((endRow - startRow) & 3);
        int fourBlocksPerRow = four(blocksPerRow);
        int fourQStride = four(qStride);
        for (; row < tailStart; row += 4) {
            dotRowsQ8PreparedBlockScalar4(
                    blocksPerRow, blockSize, quants, blockScales, matrixBlock, qBase, qStride, vector, output, row);
            matrixBlock += fourBlocksPerRow;
            qBase += fourQStride;
        }
        for (; row < endRow; row++) {
            output[row] = dotRowQ8PreparedBlockScalar(
                    blocksPerRow,
                    blockSize,
                    quants,
                    blockScales,
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
