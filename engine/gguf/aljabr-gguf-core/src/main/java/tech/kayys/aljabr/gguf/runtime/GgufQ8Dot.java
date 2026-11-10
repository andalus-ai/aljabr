package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;

/**
 * Prepared Q8-family row-dot reducers.
 *
 * <p>These reducers walk pre-unpacked signed-byte Q8 rows and combine per-block
 * scales with the low-level signed-byte dot kernels in {@link GgufDot}.</p>
 * @author bhangun
 */
final class GgufQ8Dot {
    private GgufQ8Dot() {
    }

    static float dotRowQ8Prepared32Vector(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        return dotSignedByte32ScaledBlocksVector(blocksPerRow, quants, qBase, blockScales, matrixBlock, vector);
    }

    static void dotRowsQ8Prepared32Vector4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByte32ScaledBlocksVector4(
                blocksPerRow,
                quants,
                qBase,
                qStride,
                blockScales,
                matrixBlock,
                blocksPerRow,
                vector,
                output,
                outputOffset);
    }

    static float dotRowQ8Prepared32Scalar(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 = Math.fma(
                    blockScales[matrixBlock],
                    dotSignedByte32Scalar(quants, qBase, vector, vBase),
                    sum0);
            sum1 = Math.fma(
                    blockScales[matrixBlock + 1],
                    dotSignedByte32Scalar(quants, qBase + Q4_0_BLOCK_SIZE, vector, vBase + Q4_0_BLOCK_SIZE),
                    sum1);
            sum2 = Math.fma(
                    blockScales[matrixBlock + 2],
                    dotSignedByte32Scalar(quants, qBase + 2 * Q4_0_BLOCK_SIZE, vector, vBase + 2 * Q4_0_BLOCK_SIZE),
                    sum2);
            sum3 = Math.fma(
                    blockScales[matrixBlock + 3],
                    dotSignedByte32Scalar(quants, qBase + 3 * Q4_0_BLOCK_SIZE, vector, vBase + 3 * Q4_0_BLOCK_SIZE),
                    sum3);
            matrixBlock += 4;
            qBase += 4 * Q4_0_BLOCK_SIZE;
            vBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocksPerRow; block++) {
            sum = Math.fma(blockScales[matrixBlock], dotSignedByte32Scalar(quants, qBase, vector, vBase), sum);
            matrixBlock++;
            qBase += Q4_0_BLOCK_SIZE;
            vBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static void dotRowsQ8Prepared32Scalar4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        float row0Sum0 = 0.0f;
        float row0Sum1 = 0.0f;
        float row0Sum2 = 0.0f;
        float row0Sum3 = 0.0f;
        float row1Sum0 = 0.0f;
        float row1Sum1 = 0.0f;
        float row1Sum2 = 0.0f;
        float row1Sum3 = 0.0f;
        float row2Sum0 = 0.0f;
        float row2Sum1 = 0.0f;
        float row2Sum2 = 0.0f;
        float row2Sum3 = 0.0f;
        float row3Sum0 = 0.0f;
        float row3Sum1 = 0.0f;
        float row3Sum2 = 0.0f;
        float row3Sum3 = 0.0f;
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Scale = matrixBlock;
        int row1Scale = matrixBlock + blocksPerRow;
        int row2Scale = matrixBlock + 2 * blocksPerRow;
        int row3Scale = matrixBlock + 3 * blocksPerRow;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 = Math.fma(blockScales[row0Scale], dotSignedByte32Scalar(quants, row0Q, vector, vBase), row0Sum0);
            row1Sum0 = Math.fma(blockScales[row1Scale], dotSignedByte32Scalar(quants, row1Q, vector, vBase), row1Sum0);
            row2Sum0 = Math.fma(blockScales[row2Scale], dotSignedByte32Scalar(quants, row2Q, vector, vBase), row2Sum0);
            row3Sum0 = Math.fma(blockScales[row3Scale], dotSignedByte32Scalar(quants, row3Q, vector, vBase), row3Sum0);

            int vector1 = vBase + Q4_0_BLOCK_SIZE;
            row0Sum1 = Math.fma(
                    blockScales[row0Scale + 1],
                    dotSignedByte32Scalar(quants, row0Q + Q4_0_BLOCK_SIZE, vector, vector1),
                    row0Sum1);
            row1Sum1 = Math.fma(
                    blockScales[row1Scale + 1],
                    dotSignedByte32Scalar(quants, row1Q + Q4_0_BLOCK_SIZE, vector, vector1),
                    row1Sum1);
            row2Sum1 = Math.fma(
                    blockScales[row2Scale + 1],
                    dotSignedByte32Scalar(quants, row2Q + Q4_0_BLOCK_SIZE, vector, vector1),
                    row2Sum1);
            row3Sum1 = Math.fma(
                    blockScales[row3Scale + 1],
                    dotSignedByte32Scalar(quants, row3Q + Q4_0_BLOCK_SIZE, vector, vector1),
                    row3Sum1);

            int vector2 = vBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 = Math.fma(
                    blockScales[row0Scale + 2],
                    dotSignedByte32Scalar(quants, row0Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2),
                    row0Sum2);
            row1Sum2 = Math.fma(
                    blockScales[row1Scale + 2],
                    dotSignedByte32Scalar(quants, row1Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2),
                    row1Sum2);
            row2Sum2 = Math.fma(
                    blockScales[row2Scale + 2],
                    dotSignedByte32Scalar(quants, row2Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2),
                    row2Sum2);
            row3Sum2 = Math.fma(
                    blockScales[row3Scale + 2],
                    dotSignedByte32Scalar(quants, row3Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2),
                    row3Sum2);

            int vector3 = vBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 = Math.fma(
                    blockScales[row0Scale + 3],
                    dotSignedByte32Scalar(quants, row0Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3),
                    row0Sum3);
            row1Sum3 = Math.fma(
                    blockScales[row1Scale + 3],
                    dotSignedByte32Scalar(quants, row1Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3),
                    row1Sum3);
            row2Sum3 = Math.fma(
                    blockScales[row2Scale + 3],
                    dotSignedByte32Scalar(quants, row2Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3),
                    row2Sum3);
            row3Sum3 = Math.fma(
                    blockScales[row3Scale + 3],
                    dotSignedByte32Scalar(quants, row3Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3),
                    row3Sum3);

            row0Q += 4 * Q4_0_BLOCK_SIZE;
            row1Q += 4 * Q4_0_BLOCK_SIZE;
            row2Q += 4 * Q4_0_BLOCK_SIZE;
            row3Q += 4 * Q4_0_BLOCK_SIZE;
            row0Scale += 4;
            row1Scale += 4;
            row2Scale += 4;
            row3Scale += 4;
            vBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocksPerRow; block++) {
            row0Sum = Math.fma(blockScales[row0Scale], dotSignedByte32Scalar(quants, row0Q, vector, vBase), row0Sum);
            row1Sum = Math.fma(blockScales[row1Scale], dotSignedByte32Scalar(quants, row1Q, vector, vBase), row1Sum);
            row2Sum = Math.fma(blockScales[row2Scale], dotSignedByte32Scalar(quants, row2Q, vector, vBase), row2Sum);
            row3Sum = Math.fma(blockScales[row3Scale], dotSignedByte32Scalar(quants, row3Q, vector, vBase), row3Sum);
            row0Q += Q4_0_BLOCK_SIZE;
            row1Q += Q4_0_BLOCK_SIZE;
            row2Q += Q4_0_BLOCK_SIZE;
            row3Q += Q4_0_BLOCK_SIZE;
            row0Scale++;
            row1Scale++;
            row2Scale++;
            row3Scale++;
            vBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowQ8Prepared16Vector(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        return dotSignedByte16ScaledBlocksVector(blocksPerRow, quants, qBase, blockScales, matrixBlock, vector);
    }

    static void dotRowsQ8Prepared16Vector4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByte16ScaledBlocksVector4(
                blocksPerRow,
                quants,
                qBase,
                qStride,
                blockScales,
                matrixBlock,
                blocksPerRow,
                vector,
                output,
                outputOffset);
    }

    static float dotRowQ8Prepared16Scalar(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 = Math.fma(blockScales[matrixBlock], dotSignedByte16Scalar(quants, qBase, vector, vBase), sum0);
            sum1 = Math.fma(
                    blockScales[matrixBlock + 1],
                    dotSignedByte16Scalar(quants, qBase + 16, vector, vBase + 16),
                    sum1);
            sum2 = Math.fma(
                    blockScales[matrixBlock + 2],
                    dotSignedByte16Scalar(quants, qBase + 32, vector, vBase + 32),
                    sum2);
            sum3 = Math.fma(
                    blockScales[matrixBlock + 3],
                    dotSignedByte16Scalar(quants, qBase + 48, vector, vBase + 48),
                    sum3);
            matrixBlock += 4;
            qBase += 64;
            vBase += 64;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocksPerRow; block++) {
            sum = Math.fma(blockScales[matrixBlock], dotSignedByte16Scalar(quants, qBase, vector, vBase), sum);
            matrixBlock++;
            qBase += 16;
            vBase += 16;
        }
        return sum;
    }

    static void dotRowsQ8Prepared16Scalar4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        float row0Sum0 = 0.0f;
        float row0Sum1 = 0.0f;
        float row0Sum2 = 0.0f;
        float row0Sum3 = 0.0f;
        float row1Sum0 = 0.0f;
        float row1Sum1 = 0.0f;
        float row1Sum2 = 0.0f;
        float row1Sum3 = 0.0f;
        float row2Sum0 = 0.0f;
        float row2Sum1 = 0.0f;
        float row2Sum2 = 0.0f;
        float row2Sum3 = 0.0f;
        float row3Sum0 = 0.0f;
        float row3Sum1 = 0.0f;
        float row3Sum2 = 0.0f;
        float row3Sum3 = 0.0f;
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Scale = matrixBlock;
        int row1Scale = matrixBlock + blocksPerRow;
        int row2Scale = matrixBlock + 2 * blocksPerRow;
        int row3Scale = matrixBlock + 3 * blocksPerRow;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 = Math.fma(blockScales[row0Scale], dotSignedByte16Scalar(quants, row0Q, vector, vBase), row0Sum0);
            row1Sum0 = Math.fma(blockScales[row1Scale], dotSignedByte16Scalar(quants, row1Q, vector, vBase), row1Sum0);
            row2Sum0 = Math.fma(blockScales[row2Scale], dotSignedByte16Scalar(quants, row2Q, vector, vBase), row2Sum0);
            row3Sum0 = Math.fma(blockScales[row3Scale], dotSignedByte16Scalar(quants, row3Q, vector, vBase), row3Sum0);

            int vector1 = vBase + 16;
            row0Sum1 = Math.fma(
                    blockScales[row0Scale + 1],
                    dotSignedByte16Scalar(quants, row0Q + 16, vector, vector1),
                    row0Sum1);
            row1Sum1 = Math.fma(
                    blockScales[row1Scale + 1],
                    dotSignedByte16Scalar(quants, row1Q + 16, vector, vector1),
                    row1Sum1);
            row2Sum1 = Math.fma(
                    blockScales[row2Scale + 1],
                    dotSignedByte16Scalar(quants, row2Q + 16, vector, vector1),
                    row2Sum1);
            row3Sum1 = Math.fma(
                    blockScales[row3Scale + 1],
                    dotSignedByte16Scalar(quants, row3Q + 16, vector, vector1),
                    row3Sum1);

            int vector2 = vBase + 32;
            row0Sum2 = Math.fma(
                    blockScales[row0Scale + 2],
                    dotSignedByte16Scalar(quants, row0Q + 32, vector, vector2),
                    row0Sum2);
            row1Sum2 = Math.fma(
                    blockScales[row1Scale + 2],
                    dotSignedByte16Scalar(quants, row1Q + 32, vector, vector2),
                    row1Sum2);
            row2Sum2 = Math.fma(
                    blockScales[row2Scale + 2],
                    dotSignedByte16Scalar(quants, row2Q + 32, vector, vector2),
                    row2Sum2);
            row3Sum2 = Math.fma(
                    blockScales[row3Scale + 2],
                    dotSignedByte16Scalar(quants, row3Q + 32, vector, vector2),
                    row3Sum2);

            int vector3 = vBase + 48;
            row0Sum3 = Math.fma(
                    blockScales[row0Scale + 3],
                    dotSignedByte16Scalar(quants, row0Q + 48, vector, vector3),
                    row0Sum3);
            row1Sum3 = Math.fma(
                    blockScales[row1Scale + 3],
                    dotSignedByte16Scalar(quants, row1Q + 48, vector, vector3),
                    row1Sum3);
            row2Sum3 = Math.fma(
                    blockScales[row2Scale + 3],
                    dotSignedByte16Scalar(quants, row2Q + 48, vector, vector3),
                    row2Sum3);
            row3Sum3 = Math.fma(
                    blockScales[row3Scale + 3],
                    dotSignedByte16Scalar(quants, row3Q + 48, vector, vector3),
                    row3Sum3);

            row0Q += 64;
            row1Q += 64;
            row2Q += 64;
            row3Q += 64;
            row0Scale += 4;
            row1Scale += 4;
            row2Scale += 4;
            row3Scale += 4;
            vBase += 64;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocksPerRow; block++) {
            row0Sum = Math.fma(blockScales[row0Scale], dotSignedByte16Scalar(quants, row0Q, vector, vBase), row0Sum);
            row1Sum = Math.fma(blockScales[row1Scale], dotSignedByte16Scalar(quants, row1Q, vector, vBase), row1Sum);
            row2Sum = Math.fma(blockScales[row2Scale], dotSignedByte16Scalar(quants, row2Q, vector, vBase), row2Sum);
            row3Sum = Math.fma(blockScales[row3Scale], dotSignedByte16Scalar(quants, row3Q, vector, vBase), row3Sum);
            row0Q += 16;
            row1Q += 16;
            row2Q += 16;
            row3Q += 16;
            row0Scale++;
            row1Scale++;
            row2Scale++;
            row3Scale++;
            vBase += 16;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowQ8PreparedWideVector(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        return dotSignedByteWideScaledBlocksVector(
                blocksPerRow, blockSize, quants, qBase, blockScales, matrixBlock, vector);
    }

    static void dotRowsQ8PreparedWideVector4(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByteWideScaledBlocksVector4(
                blocksPerRow,
                blockSize,
                quants,
                qBase,
                qStride,
                blockScales,
                matrixBlock,
                blocksPerRow,
                vector,
                output,
                outputOffset);
    }

    static float dotRowQ8PreparedWideScalar(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 = Math.fma(
                    blockScales[matrixBlock],
                    dotSignedByteWideBlockScalar(quants, qBase, vector, vBase, blockSize),
                    sum0);
            sum1 = Math.fma(
                    blockScales[matrixBlock + 1],
                    dotSignedByteWideBlockScalar(quants, qBase + blockSize, vector, vBase + blockSize, blockSize),
                    sum1);
            sum2 = Math.fma(
                    blockScales[matrixBlock + 2],
                    dotSignedByteWideBlockScalar(quants, qBase + 2 * blockSize, vector, vBase + 2 * blockSize, blockSize),
                    sum2);
            sum3 = Math.fma(
                    blockScales[matrixBlock + 3],
                    dotSignedByteWideBlockScalar(quants, qBase + 3 * blockSize, vector, vBase + 3 * blockSize, blockSize),
                    sum3);
            matrixBlock += 4;
            qBase += 4 * blockSize;
            vBase += 4 * blockSize;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocksPerRow; block++) {
            sum = Math.fma(
                    blockScales[matrixBlock],
                    dotSignedByteWideBlockScalar(quants, qBase, vector, vBase, blockSize),
                    sum);
            matrixBlock++;
            qBase += blockSize;
            vBase += blockSize;
        }
        return sum;
    }

    static void dotRowsQ8PreparedWideScalar4(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotRowsQ8PreparedBlockScalar4(
                blocksPerRow, blockSize, quants, blockScales, matrixBlock, qBase, qStride, vector, output, outputOffset);
    }

    static float dotRowQ8PreparedBlockVector(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        return dotSignedByteScaledBlocksVector(
                blocksPerRow, blockSize, quants, qBase, blockScales, matrixBlock, vector);
    }

    static void dotRowsQ8PreparedBlockVector4(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByteScaledBlocksVector4(
                blocksPerRow,
                blockSize,
                quants,
                qBase,
                qStride,
                blockScales,
                matrixBlock,
                blocksPerRow,
                vector,
                output,
                outputOffset);
    }

    static float dotRowQ8PreparedBlockScalar(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 = Math.fma(
                    blockScales[matrixBlock],
                    dotSignedByteBlockScalar(quants, qBase, vector, vBase, blockSize),
                    sum0);
            sum1 = Math.fma(
                    blockScales[matrixBlock + 1],
                    dotSignedByteBlockScalar(quants, qBase + blockSize, vector, vBase + blockSize, blockSize),
                    sum1);
            sum2 = Math.fma(
                    blockScales[matrixBlock + 2],
                    dotSignedByteBlockScalar(quants, qBase + 2 * blockSize, vector, vBase + 2 * blockSize, blockSize),
                    sum2);
            sum3 = Math.fma(
                    blockScales[matrixBlock + 3],
                    dotSignedByteBlockScalar(quants, qBase + 3 * blockSize, vector, vBase + 3 * blockSize, blockSize),
                    sum3);
            matrixBlock += 4;
            qBase += 4 * blockSize;
            vBase += 4 * blockSize;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocksPerRow; block++) {
            sum = Math.fma(
                    blockScales[matrixBlock],
                    dotSignedByteBlockScalar(quants, qBase, vector, vBase, blockSize),
                    sum);
            matrixBlock++;
            qBase += blockSize;
            vBase += blockSize;
        }
        return sum;
    }

    static void dotRowsQ8PreparedBlockScalar4(
            int blocksPerRow,
            int blockSize,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        float row0Sum0 = 0.0f;
        float row0Sum1 = 0.0f;
        float row0Sum2 = 0.0f;
        float row0Sum3 = 0.0f;
        float row1Sum0 = 0.0f;
        float row1Sum1 = 0.0f;
        float row1Sum2 = 0.0f;
        float row1Sum3 = 0.0f;
        float row2Sum0 = 0.0f;
        float row2Sum1 = 0.0f;
        float row2Sum2 = 0.0f;
        float row2Sum3 = 0.0f;
        float row3Sum0 = 0.0f;
        float row3Sum1 = 0.0f;
        float row3Sum2 = 0.0f;
        float row3Sum3 = 0.0f;
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Scale = matrixBlock;
        int row1Scale = matrixBlock + blocksPerRow;
        int row2Scale = matrixBlock + 2 * blocksPerRow;
        int row3Scale = matrixBlock + 3 * blocksPerRow;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        int twoBlockSize = blockSize + blockSize;
        int threeBlockSize = twoBlockSize + blockSize;
        int fourBlockSize = threeBlockSize + blockSize;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 = Math.fma(
                    blockScales[row0Scale],
                    dotSignedByteBlockScalar(quants, row0Q, vector, vBase, blockSize),
                    row0Sum0);
            row1Sum0 = Math.fma(
                    blockScales[row1Scale],
                    dotSignedByteBlockScalar(quants, row1Q, vector, vBase, blockSize),
                    row1Sum0);
            row2Sum0 = Math.fma(
                    blockScales[row2Scale],
                    dotSignedByteBlockScalar(quants, row2Q, vector, vBase, blockSize),
                    row2Sum0);
            row3Sum0 = Math.fma(
                    blockScales[row3Scale],
                    dotSignedByteBlockScalar(quants, row3Q, vector, vBase, blockSize),
                    row3Sum0);

            int vector1 = vBase + blockSize;
            row0Sum1 = Math.fma(
                    blockScales[row0Scale + 1],
                    dotSignedByteBlockScalar(quants, row0Q + blockSize, vector, vector1, blockSize),
                    row0Sum1);
            row1Sum1 = Math.fma(
                    blockScales[row1Scale + 1],
                    dotSignedByteBlockScalar(quants, row1Q + blockSize, vector, vector1, blockSize),
                    row1Sum1);
            row2Sum1 = Math.fma(
                    blockScales[row2Scale + 1],
                    dotSignedByteBlockScalar(quants, row2Q + blockSize, vector, vector1, blockSize),
                    row2Sum1);
            row3Sum1 = Math.fma(
                    blockScales[row3Scale + 1],
                    dotSignedByteBlockScalar(quants, row3Q + blockSize, vector, vector1, blockSize),
                    row3Sum1);

            int vector2 = vBase + twoBlockSize;
            row0Sum2 = Math.fma(
                    blockScales[row0Scale + 2],
                    dotSignedByteBlockScalar(quants, row0Q + twoBlockSize, vector, vector2, blockSize),
                    row0Sum2);
            row1Sum2 = Math.fma(
                    blockScales[row1Scale + 2],
                    dotSignedByteBlockScalar(quants, row1Q + twoBlockSize, vector, vector2, blockSize),
                    row1Sum2);
            row2Sum2 = Math.fma(
                    blockScales[row2Scale + 2],
                    dotSignedByteBlockScalar(quants, row2Q + twoBlockSize, vector, vector2, blockSize),
                    row2Sum2);
            row3Sum2 = Math.fma(
                    blockScales[row3Scale + 2],
                    dotSignedByteBlockScalar(quants, row3Q + twoBlockSize, vector, vector2, blockSize),
                    row3Sum2);

            int vector3 = vBase + threeBlockSize;
            row0Sum3 = Math.fma(
                    blockScales[row0Scale + 3],
                    dotSignedByteBlockScalar(quants, row0Q + threeBlockSize, vector, vector3, blockSize),
                    row0Sum3);
            row1Sum3 = Math.fma(
                    blockScales[row1Scale + 3],
                    dotSignedByteBlockScalar(quants, row1Q + threeBlockSize, vector, vector3, blockSize),
                    row1Sum3);
            row2Sum3 = Math.fma(
                    blockScales[row2Scale + 3],
                    dotSignedByteBlockScalar(quants, row2Q + threeBlockSize, vector, vector3, blockSize),
                    row2Sum3);
            row3Sum3 = Math.fma(
                    blockScales[row3Scale + 3],
                    dotSignedByteBlockScalar(quants, row3Q + threeBlockSize, vector, vector3, blockSize),
                    row3Sum3);

            row0Q += fourBlockSize;
            row1Q += fourBlockSize;
            row2Q += fourBlockSize;
            row3Q += fourBlockSize;
            row0Scale += 4;
            row1Scale += 4;
            row2Scale += 4;
            row3Scale += 4;
            vBase += fourBlockSize;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocksPerRow; block++) {
            row0Sum = Math.fma(
                    blockScales[row0Scale],
                    dotSignedByteBlockScalar(quants, row0Q, vector, vBase, blockSize),
                    row0Sum);
            row1Sum = Math.fma(
                    blockScales[row1Scale],
                    dotSignedByteBlockScalar(quants, row1Q, vector, vBase, blockSize),
                    row1Sum);
            row2Sum = Math.fma(
                    blockScales[row2Scale],
                    dotSignedByteBlockScalar(quants, row2Q, vector, vBase, blockSize),
                    row2Sum);
            row3Sum = Math.fma(
                    blockScales[row3Scale],
                    dotSignedByteBlockScalar(quants, row3Q, vector, vBase, blockSize),
                    row3Sum);
            row0Q += blockSize;
            row1Q += blockSize;
            row2Q += blockSize;
            row3Q += blockSize;
            row0Scale++;
            row1Scale++;
            row2Scale++;
            row3Scale++;
            vBase += blockSize;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }
}
