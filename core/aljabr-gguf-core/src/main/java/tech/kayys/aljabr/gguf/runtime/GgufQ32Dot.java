package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufDot.dotSignedByte32Scalar;
import static tech.kayys.aljabr.gguf.runtime.GgufDot.dotSignedByte32AffineScalar;
import static tech.kayys.aljabr.gguf.runtime.GgufDot.dotSignedByte32AffineBlocksVector;
import static tech.kayys.aljabr.gguf.runtime.GgufDot.dotSignedByte32AffineBlocksVector4;
import static tech.kayys.aljabr.gguf.runtime.GgufDot.dotSignedByte32ScaledBlocksVector;
import static tech.kayys.aljabr.gguf.runtime.GgufDot.dotSignedByte32ScaledBlocksVector4;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_VECTOR_LANES;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Prepared Q32-family row-dot reducers.
 *
 * <p>Q4_0/Q5_0-style prepared matrices share 32-value blocks. This helper owns
 * the bias and no-bias reducers so the tensor facade only coordinates row
 * traversal and scheduling.</p>
 * @author bhangun
 */
final class GgufQ32Dot {
    private GgufQ32Dot() {
    }

    static float dotRowQ32PreparedVector(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            int qBase,
            float[] vector,
            float[] vectorGroupSums) {
        float sum = dotSignedByte32ScaledBlocksVector(blocksPerRow, quants, qBase, blockScales, matrixBlock, vector);
        return sum + dotBlockBiasesVector(blocksPerRow, blockBiases, matrixBlock, vectorGroupSums);
    }

    static float dotRowQ32PreparedDirectVector(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            int qBase,
            float[] vector) {
        return dotSignedByte32AffineBlocksVector(
                blocksPerRow, quants, qBase, blockScales, blockBiases, matrixBlock, 1.0f, vector);
    }

    static void dotRowsQ32PreparedDirectVector4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            int blockStride,
            int qBase,
            int qStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByte32AffineBlocksVector4(
                blocksPerRow,
                quants,
                qBase,
                qStride,
                blockScales,
                blockBiases,
                matrixBlock,
                blockStride,
                1.0f,
                vector,
                output,
                outputOffset);
    }

    static void dotRowsQ32PreparedVector4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            int blockStride,
            int qBase,
            int qStride,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int outputOffset) {
        dotSignedByte32ScaledBlocksVector4(
                blocksPerRow,
                quants,
                qBase,
                qStride,
                blockScales,
                matrixBlock,
                blockStride,
                vector,
                output,
                outputOffset);
        addBlockBiasesVector4(
                blocksPerRow,
                blockBiases,
                matrixBlock,
                blockStride,
                vectorGroupSums,
                output,
                outputOffset);
    }

    static void dotRowsQ32PreparedScalar4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            int blockStride,
            int qBase,
            int qStride,
            float[] vector,
            float[] vectorGroupSums,
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
        int row0Block = matrixBlock;
        int row1Block = matrixBlock + blockStride;
        int row2Block = matrixBlock + 2 * blockStride;
        int row3Block = matrixBlock + 3 * blockStride;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            float vectorSum0 = vectorGroupSums[block];
            row0Sum0 += q32PrecomputedBlock(quants, row0Q, vector, vBase, blockScales, blockBiases, row0Block, vectorSum0);
            row1Sum0 += q32PrecomputedBlock(quants, row1Q, vector, vBase, blockScales, blockBiases, row1Block, vectorSum0);
            row2Sum0 += q32PrecomputedBlock(quants, row2Q, vector, vBase, blockScales, blockBiases, row2Block, vectorSum0);
            row3Sum0 += q32PrecomputedBlock(quants, row3Q, vector, vBase, blockScales, blockBiases, row3Block, vectorSum0);

            int vector1 = vBase + Q4_0_BLOCK_SIZE;
            float vectorSum1 = vectorGroupSums[block + 1];
            row0Sum1 += q32PrecomputedBlock(
                    quants, row0Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row0Block + 1, vectorSum1);
            row1Sum1 += q32PrecomputedBlock(
                    quants, row1Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row1Block + 1, vectorSum1);
            row2Sum1 += q32PrecomputedBlock(
                    quants, row2Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row2Block + 1, vectorSum1);
            row3Sum1 += q32PrecomputedBlock(
                    quants, row3Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row3Block + 1, vectorSum1);

            int vector2 = vBase + 2 * Q4_0_BLOCK_SIZE;
            float vectorSum2 = vectorGroupSums[block + 2];
            row0Sum2 += q32PrecomputedBlock(
                    quants, row0Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row0Block + 2, vectorSum2);
            row1Sum2 += q32PrecomputedBlock(
                    quants, row1Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row1Block + 2, vectorSum2);
            row2Sum2 += q32PrecomputedBlock(
                    quants, row2Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row2Block + 2, vectorSum2);
            row3Sum2 += q32PrecomputedBlock(
                    quants, row3Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row3Block + 2, vectorSum2);

            int vector3 = vBase + 3 * Q4_0_BLOCK_SIZE;
            float vectorSum3 = vectorGroupSums[block + 3];
            row0Sum3 += q32PrecomputedBlock(
                    quants, row0Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row0Block + 3, vectorSum3);
            row1Sum3 += q32PrecomputedBlock(
                    quants, row1Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row1Block + 3, vectorSum3);
            row2Sum3 += q32PrecomputedBlock(
                    quants, row2Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row2Block + 3, vectorSum3);
            row3Sum3 += q32PrecomputedBlock(
                    quants, row3Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row3Block + 3, vectorSum3);

            row0Q += 4 * Q4_0_BLOCK_SIZE;
            row1Q += 4 * Q4_0_BLOCK_SIZE;
            row2Q += 4 * Q4_0_BLOCK_SIZE;
            row3Q += 4 * Q4_0_BLOCK_SIZE;
            row0Block += 4;
            row1Block += 4;
            row2Block += 4;
            row3Block += 4;
            vBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocksPerRow; block++) {
            float vectorSum = vectorGroupSums[block];
            row0Sum += q32PrecomputedBlock(quants, row0Q, vector, vBase, blockScales, blockBiases, row0Block, vectorSum);
            row1Sum += q32PrecomputedBlock(quants, row1Q, vector, vBase, blockScales, blockBiases, row1Block, vectorSum);
            row2Sum += q32PrecomputedBlock(quants, row2Q, vector, vBase, blockScales, blockBiases, row2Block, vectorSum);
            row3Sum += q32PrecomputedBlock(quants, row3Q, vector, vBase, blockScales, blockBiases, row3Block, vectorSum);
            row0Q += Q4_0_BLOCK_SIZE;
            row1Q += Q4_0_BLOCK_SIZE;
            row2Q += Q4_0_BLOCK_SIZE;
            row3Q += Q4_0_BLOCK_SIZE;
            row0Block++;
            row1Block++;
            row2Block++;
            row3Block++;
            vBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float q32PrecomputedBlock(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            float vectorSum) {
        return blockScales[matrixBlock] * dotSignedByte32Scalar(quants, qBase, vector, vBase)
                + blockBiases[matrixBlock] * vectorSum;
    }

    static float dotRowQ32PreparedScalar(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            int qBase,
            float[] vector,
            float[] vectorGroupSums) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += blockScales[matrixBlock] * dotSignedByte32Scalar(quants, qBase, vector, vBase)
                    + blockBiases[matrixBlock] * vectorGroupSums[block];
            sum1 += blockScales[matrixBlock + 1]
                            * dotSignedByte32Scalar(
                                    quants, qBase + Q4_0_BLOCK_SIZE, vector, vBase + Q4_0_BLOCK_SIZE)
                    + blockBiases[matrixBlock + 1] * vectorGroupSums[block + 1];
            sum2 += blockScales[matrixBlock + 2]
                            * dotSignedByte32Scalar(
                                    quants, qBase + 2 * Q4_0_BLOCK_SIZE, vector, vBase + 2 * Q4_0_BLOCK_SIZE)
                    + blockBiases[matrixBlock + 2] * vectorGroupSums[block + 2];
            sum3 += blockScales[matrixBlock + 3]
                            * dotSignedByte32Scalar(
                                    quants, qBase + 3 * Q4_0_BLOCK_SIZE, vector, vBase + 3 * Q4_0_BLOCK_SIZE)
                    + blockBiases[matrixBlock + 3] * vectorGroupSums[block + 3];
            matrixBlock += 4;
            qBase += 4 * Q4_0_BLOCK_SIZE;
            vBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocksPerRow; block++) {
            float quantDot = dotSignedByte32Scalar(quants, qBase, vector, vBase);
            float scale = blockScales[matrixBlock];
            float bias = blockBiases[matrixBlock];
            float vectorSum = vectorGroupSums[block];
            sum += scale * quantDot + bias * vectorSum;
            matrixBlock++;
            qBase += Q4_0_BLOCK_SIZE;
            vBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static float dotBlockBiasesVector(
            int blocks,
            float[] blockBiases,
            int biasBase,
            float[] vectorGroupSums) {
        FloatVector acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        int block = 0;
        int unrolledStride = FLOAT_SUM_VECTOR_LANES * 4;
        int unrolledLimit = blocks - unrolledStride;
        for (; block <= unrolledLimit; block += unrolledStride) {
            acc0 = FloatVector.fromArray(FLOAT_SUM_SPECIES, blockBiases, biasBase + block)
                    .fma(FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, block), acc0);
            acc1 = FloatVector.fromArray(FLOAT_SUM_SPECIES, blockBiases, biasBase + block + FLOAT_SUM_VECTOR_LANES)
                    .fma(FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, vectorGroupSums, block + FLOAT_SUM_VECTOR_LANES), acc1);
            acc2 = FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, blockBiases, biasBase + block + 2 * FLOAT_SUM_VECTOR_LANES)
                    .fma(FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, vectorGroupSums, block + 2 * FLOAT_SUM_VECTOR_LANES), acc2);
            acc3 = FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, blockBiases, biasBase + block + 3 * FLOAT_SUM_VECTOR_LANES)
                    .fma(FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, vectorGroupSums, block + 3 * FLOAT_SUM_VECTOR_LANES), acc3);
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        int vectorLimit = blocks - FLOAT_SUM_VECTOR_LANES;
        for (; block <= vectorLimit; block += FLOAT_SUM_VECTOR_LANES) {
            acc = FloatVector.fromArray(FLOAT_SUM_SPECIES, blockBiases, biasBase + block)
                    .fma(FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, block), acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; block < blocks; block++) {
            sum = Math.fma(blockBiases[biasBase + block], vectorGroupSums[block], sum);
        }
        return sum;
    }

    private static void addBlockBiasesVector4(
            int blocks,
            float[] blockBiases,
            int biasBase,
            int biasStride,
            float[] vectorGroupSums,
            float[] output,
            int outputOffset) {
        int row0Base = biasBase;
        int row1Base = biasBase + biasStride;
        int row2Base = biasBase + 2 * biasStride;
        int row3Base = biasBase + 3 * biasStride;
        FloatVector row0Acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row0Acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row0Acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row0Acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row1Acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row1Acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row1Acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row1Acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row2Acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row2Acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row2Acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row2Acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row3Acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row3Acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row3Acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector row3Acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        int block = 0;
        int unrolledStride = FLOAT_SUM_VECTOR_LANES * 4;
        int unrolledLimit = blocks - unrolledStride;
        for (; block <= unrolledLimit; block += unrolledStride) {
            FloatVector sums0 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, block);
            row0Acc0 = accumulateBlockBiasesVector(blockBiases, row0Base + block, sums0, row0Acc0);
            row1Acc0 = accumulateBlockBiasesVector(blockBiases, row1Base + block, sums0, row1Acc0);
            row2Acc0 = accumulateBlockBiasesVector(blockBiases, row2Base + block, sums0, row2Acc0);
            row3Acc0 = accumulateBlockBiasesVector(blockBiases, row3Base + block, sums0, row3Acc0);

            int block1 = block + FLOAT_SUM_VECTOR_LANES;
            FloatVector sums1 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, block1);
            row0Acc1 = accumulateBlockBiasesVector(blockBiases, row0Base + block1, sums1, row0Acc1);
            row1Acc1 = accumulateBlockBiasesVector(blockBiases, row1Base + block1, sums1, row1Acc1);
            row2Acc1 = accumulateBlockBiasesVector(blockBiases, row2Base + block1, sums1, row2Acc1);
            row3Acc1 = accumulateBlockBiasesVector(blockBiases, row3Base + block1, sums1, row3Acc1);

            int block2 = block + 2 * FLOAT_SUM_VECTOR_LANES;
            FloatVector sums2 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, block2);
            row0Acc2 = accumulateBlockBiasesVector(blockBiases, row0Base + block2, sums2, row0Acc2);
            row1Acc2 = accumulateBlockBiasesVector(blockBiases, row1Base + block2, sums2, row1Acc2);
            row2Acc2 = accumulateBlockBiasesVector(blockBiases, row2Base + block2, sums2, row2Acc2);
            row3Acc2 = accumulateBlockBiasesVector(blockBiases, row3Base + block2, sums2, row3Acc2);

            int block3 = block + 3 * FLOAT_SUM_VECTOR_LANES;
            FloatVector sums3 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, block3);
            row0Acc3 = accumulateBlockBiasesVector(blockBiases, row0Base + block3, sums3, row0Acc3);
            row1Acc3 = accumulateBlockBiasesVector(blockBiases, row1Base + block3, sums3, row1Acc3);
            row2Acc3 = accumulateBlockBiasesVector(blockBiases, row2Base + block3, sums3, row2Acc3);
            row3Acc3 = accumulateBlockBiasesVector(blockBiases, row3Base + block3, sums3, row3Acc3);
        }
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        int vectorLimit = blocks - FLOAT_SUM_VECTOR_LANES;
        for (; block <= vectorLimit; block += FLOAT_SUM_VECTOR_LANES) {
            FloatVector sums = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, block);
            row0Acc = accumulateBlockBiasesVector(blockBiases, row0Base + block, sums, row0Acc);
            row1Acc = accumulateBlockBiasesVector(blockBiases, row1Base + block, sums, row1Acc);
            row2Acc = accumulateBlockBiasesVector(blockBiases, row2Base + block, sums, row2Acc);
            row3Acc = accumulateBlockBiasesVector(blockBiases, row3Base + block, sums, row3Acc);
        }
        float row0Sum = row0Acc.reduceLanes(VectorOperators.ADD);
        float row1Sum = row1Acc.reduceLanes(VectorOperators.ADD);
        float row2Sum = row2Acc.reduceLanes(VectorOperators.ADD);
        float row3Sum = row3Acc.reduceLanes(VectorOperators.ADD);
        for (; block < blocks; block++) {
            float vectorSum = vectorGroupSums[block];
            row0Sum = Math.fma(blockBiases[row0Base + block], vectorSum, row0Sum);
            row1Sum = Math.fma(blockBiases[row1Base + block], vectorSum, row1Sum);
            row2Sum = Math.fma(blockBiases[row2Base + block], vectorSum, row2Sum);
            row3Sum = Math.fma(blockBiases[row3Base + block], vectorSum, row3Sum);
        }
        output[outputOffset] += row0Sum;
        output[outputOffset + 1] += row1Sum;
        output[outputOffset + 2] += row2Sum;
        output[outputOffset + 3] += row3Sum;
    }

    private static FloatVector accumulateBlockBiasesVector(
            float[] blockBiases,
            int biasBase,
            FloatVector vectorGroupSums,
            FloatVector acc) {
        return FloatVector.fromArray(FLOAT_SUM_SPECIES, blockBiases, biasBase).fma(vectorGroupSums, acc);
    }

    static void dotRowsQ32PreparedDirectScalar4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock,
            int blockStride,
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
        int row0Block = matrixBlock;
        int row1Block = matrixBlock + blockStride;
        int row2Block = matrixBlock + 2 * blockStride;
        int row3Block = matrixBlock + 3 * blockStride;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += q32DirectBlock(quants, row0Q, vector, vBase, blockScales, blockBiases, row0Block);
            row1Sum0 += q32DirectBlock(quants, row1Q, vector, vBase, blockScales, blockBiases, row1Block);
            row2Sum0 += q32DirectBlock(quants, row2Q, vector, vBase, blockScales, blockBiases, row2Block);
            row3Sum0 += q32DirectBlock(quants, row3Q, vector, vBase, blockScales, blockBiases, row3Block);

            int vector1 = vBase + Q4_0_BLOCK_SIZE;
            row0Sum1 += q32DirectBlock(quants, row0Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row0Block + 1);
            row1Sum1 += q32DirectBlock(quants, row1Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row1Block + 1);
            row2Sum1 += q32DirectBlock(quants, row2Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row2Block + 1);
            row3Sum1 += q32DirectBlock(quants, row3Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, blockBiases, row3Block + 1);

            int vector2 = vBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 += q32DirectBlock(quants, row0Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row0Block + 2);
            row1Sum2 += q32DirectBlock(quants, row1Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row1Block + 2);
            row2Sum2 += q32DirectBlock(quants, row2Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row2Block + 2);
            row3Sum2 += q32DirectBlock(quants, row3Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, blockBiases, row3Block + 2);

            int vector3 = vBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 += q32DirectBlock(quants, row0Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row0Block + 3);
            row1Sum3 += q32DirectBlock(quants, row1Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row1Block + 3);
            row2Sum3 += q32DirectBlock(quants, row2Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row2Block + 3);
            row3Sum3 += q32DirectBlock(quants, row3Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, blockBiases, row3Block + 3);

            row0Q += 4 * Q4_0_BLOCK_SIZE;
            row1Q += 4 * Q4_0_BLOCK_SIZE;
            row2Q += 4 * Q4_0_BLOCK_SIZE;
            row3Q += 4 * Q4_0_BLOCK_SIZE;
            row0Block += 4;
            row1Block += 4;
            row2Block += 4;
            row3Block += 4;
            vBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocksPerRow; block++) {
            row0Sum += q32DirectBlock(quants, row0Q, vector, vBase, blockScales, blockBiases, row0Block);
            row1Sum += q32DirectBlock(quants, row1Q, vector, vBase, blockScales, blockBiases, row1Block);
            row2Sum += q32DirectBlock(quants, row2Q, vector, vBase, blockScales, blockBiases, row2Block);
            row3Sum += q32DirectBlock(quants, row3Q, vector, vBase, blockScales, blockBiases, row3Block);
            row0Q += Q4_0_BLOCK_SIZE;
            row1Q += Q4_0_BLOCK_SIZE;
            row2Q += Q4_0_BLOCK_SIZE;
            row3Q += Q4_0_BLOCK_SIZE;
            row0Block++;
            row1Block++;
            row2Block++;
            row3Block++;
            vBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float q32DirectBlock(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] blockScales,
            float[] blockBiases,
            int matrixBlock) {
        return dotSignedByte32AffineScalar(
                quants, qBase, vector, vBase, blockScales[matrixBlock], blockBiases[matrixBlock]);
    }

    static float dotRowQ32PreparedDirectScalar(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
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
            sum0 += dotSignedByte32AffineScalar(
                    quants, qBase, vector, vBase, blockScales[matrixBlock], blockBiases[matrixBlock]);
            sum1 += dotSignedByte32AffineScalar(
                    quants,
                    qBase + Q4_0_BLOCK_SIZE,
                    vector,
                    vBase + Q4_0_BLOCK_SIZE,
                    blockScales[matrixBlock + 1],
                    blockBiases[matrixBlock + 1]);
            sum2 += dotSignedByte32AffineScalar(
                    quants,
                    qBase + 2 * Q4_0_BLOCK_SIZE,
                    vector,
                    vBase + 2 * Q4_0_BLOCK_SIZE,
                    blockScales[matrixBlock + 2],
                    blockBiases[matrixBlock + 2]);
            sum3 += dotSignedByte32AffineScalar(
                    quants,
                    qBase + 3 * Q4_0_BLOCK_SIZE,
                    vector,
                    vBase + 3 * Q4_0_BLOCK_SIZE,
                    blockScales[matrixBlock + 3],
                    blockBiases[matrixBlock + 3]);
            matrixBlock += 4;
            qBase += 4 * Q4_0_BLOCK_SIZE;
            vBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocksPerRow; block++) {
            float scale = blockScales[matrixBlock];
            float bias = blockBiases[matrixBlock];
            sum += dotSignedByte32AffineScalar(quants, qBase, vector, vBase, scale, bias);
            matrixBlock++;
            qBase += Q4_0_BLOCK_SIZE;
            vBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ32PreparedNoBiasVector(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int qBase,
            float[] vector) {
        return dotSignedByte32ScaledBlocksVector(blocksPerRow, quants, qBase, blockScales, matrixBlock, vector);
    }

    static void dotRowsQ32PreparedNoBiasVector4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int blockStride,
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
                blockStride,
                vector,
                output,
                outputOffset);
    }

    static float dotRowQ32PreparedNoBiasScalar(
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

    static void dotRowsQ32PreparedNoBiasScalar4(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int matrixBlock,
            int blockStride,
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
        int row0Block = matrixBlock;
        int row1Block = matrixBlock + blockStride;
        int row2Block = matrixBlock + 2 * blockStride;
        int row3Block = matrixBlock + 3 * blockStride;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocksPerRow - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 = q32NoBiasBlock(row0Sum0, quants, row0Q, vector, vBase, blockScales, row0Block);
            row1Sum0 = q32NoBiasBlock(row1Sum0, quants, row1Q, vector, vBase, blockScales, row1Block);
            row2Sum0 = q32NoBiasBlock(row2Sum0, quants, row2Q, vector, vBase, blockScales, row2Block);
            row3Sum0 = q32NoBiasBlock(row3Sum0, quants, row3Q, vector, vBase, blockScales, row3Block);

            int vector1 = vBase + Q4_0_BLOCK_SIZE;
            row0Sum1 = q32NoBiasBlock(row0Sum1, quants, row0Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, row0Block + 1);
            row1Sum1 = q32NoBiasBlock(row1Sum1, quants, row1Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, row1Block + 1);
            row2Sum1 = q32NoBiasBlock(row2Sum1, quants, row2Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, row2Block + 1);
            row3Sum1 = q32NoBiasBlock(row3Sum1, quants, row3Q + Q4_0_BLOCK_SIZE, vector, vector1, blockScales, row3Block + 1);

            int vector2 = vBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 = q32NoBiasBlock(row0Sum2, quants, row0Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, row0Block + 2);
            row1Sum2 = q32NoBiasBlock(row1Sum2, quants, row1Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, row1Block + 2);
            row2Sum2 = q32NoBiasBlock(row2Sum2, quants, row2Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, row2Block + 2);
            row3Sum2 = q32NoBiasBlock(row3Sum2, quants, row3Q + 2 * Q4_0_BLOCK_SIZE, vector, vector2, blockScales, row3Block + 2);

            int vector3 = vBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 = q32NoBiasBlock(row0Sum3, quants, row0Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, row0Block + 3);
            row1Sum3 = q32NoBiasBlock(row1Sum3, quants, row1Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, row1Block + 3);
            row2Sum3 = q32NoBiasBlock(row2Sum3, quants, row2Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, row2Block + 3);
            row3Sum3 = q32NoBiasBlock(row3Sum3, quants, row3Q + 3 * Q4_0_BLOCK_SIZE, vector, vector3, blockScales, row3Block + 3);

            row0Q += 4 * Q4_0_BLOCK_SIZE;
            row1Q += 4 * Q4_0_BLOCK_SIZE;
            row2Q += 4 * Q4_0_BLOCK_SIZE;
            row3Q += 4 * Q4_0_BLOCK_SIZE;
            row0Block += 4;
            row1Block += 4;
            row2Block += 4;
            row3Block += 4;
            vBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocksPerRow; block++) {
            row0Sum = q32NoBiasBlock(row0Sum, quants, row0Q, vector, vBase, blockScales, row0Block);
            row1Sum = q32NoBiasBlock(row1Sum, quants, row1Q, vector, vBase, blockScales, row1Block);
            row2Sum = q32NoBiasBlock(row2Sum, quants, row2Q, vector, vBase, blockScales, row2Block);
            row3Sum = q32NoBiasBlock(row3Sum, quants, row3Q, vector, vBase, blockScales, row3Block);
            row0Q += Q4_0_BLOCK_SIZE;
            row1Q += Q4_0_BLOCK_SIZE;
            row2Q += Q4_0_BLOCK_SIZE;
            row3Q += Q4_0_BLOCK_SIZE;
            row0Block++;
            row1Block++;
            row2Block++;
            row3Block++;
            vBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float q32NoBiasBlock(
            float sum,
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] blockScales,
            int matrixBlock) {
        return Math.fma(blockScales[matrixBlock], dotSignedByte32Scalar(quants, qBase, vector, vBase), sum);
    }
}
