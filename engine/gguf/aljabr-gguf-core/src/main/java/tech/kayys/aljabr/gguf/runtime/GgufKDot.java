package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_VECTOR_LANES;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Prepared K-quant row-dot reducers.
 *
 * <p>K-family prepared rows use 16-value or 32-value groups with optional
 * per-group mins. This helper keeps those group reducers separate from row
 * scheduling and from the lower-level signed-byte dot kernels.</p>
 * @author bhangun
 */
final class GgufKDot {
    private GgufKDot() {
    }

    static float dotRowQ2KPreparedVector(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector,
            float[] vectorGroupSums) {
        float sum = dotSignedByte16ScaledBlocksVector(groups, quants, quantsOffset, groupScales, matrixGroupBase, vector);
        return sum - dotGroupMinsVector(groups, groupMins, matrixGroupBase, vectorGroupSums);
    }

    static void dotRowsQ2KPreparedVector4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int outputOffset) {
        dotSignedByte16ScaledBlocksVector4(
                groups,
                quants,
                quantsOffset,
                rowQuantStride,
                groupScales,
                matrixGroupBase,
                rowGroupStride,
                vector,
                output,
                outputOffset);
        subtractGroupMinsVector4(groups, groupMins, matrixGroupBase, rowGroupStride, vectorGroupSums, output,
                outputOffset);
    }

    static float dotRowQ2KPreparedDirectVector(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        return dotSignedByte16AffineBlocksVector(
                groups, quants, quantsOffset, groupScales, groupMins, matrixGroupBase, -1.0f, vector);
    }

    static void dotRowsQ2KPreparedDirectVector4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByte16AffineBlocksVector4(
                groups,
                quants,
                quantsOffset,
                rowQuantStride,
                groupScales,
                groupMins,
                matrixGroupBase,
                rowGroupStride,
                -1.0f,
                vector,
                output,
                outputOffset);
    }

    static float dotRowQ2KPreparedScalar(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector,
            float[] vectorGroupSums) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int qBase = quantsOffset;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            sum0 += groupScales[matrixGroupBase] * dotSignedByte16Scalar(quants, qBase, vector, vBase)
                    - groupMins[matrixGroupBase] * vectorGroupSums[group];
            sum1 += groupScales[matrixGroupBase + 1]
                            * dotSignedByte16Scalar(quants, qBase + 16, vector, vBase + 16)
                    - groupMins[matrixGroupBase + 1] * vectorGroupSums[group + 1];
            sum2 += groupScales[matrixGroupBase + 2]
                            * dotSignedByte16Scalar(quants, qBase + 32, vector, vBase + 32)
                    - groupMins[matrixGroupBase + 2] * vectorGroupSums[group + 2];
            sum3 += groupScales[matrixGroupBase + 3]
                            * dotSignedByte16Scalar(quants, qBase + 48, vector, vBase + 48)
                    - groupMins[matrixGroupBase + 3] * vectorGroupSums[group + 3];
            matrixGroupBase += 4;
            qBase += 64;
            vBase += 64;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; group < groups; group++) {
            float quantDot = dotSignedByte16Scalar(quants, qBase, vector, vBase);
            float scale = groupScales[matrixGroupBase];
            float min = groupMins[matrixGroupBase];
            float vectorSum = vectorGroupSums[group];
            sum += scale * quantDot - min * vectorSum;
            matrixGroupBase++;
            qBase += 16;
            vBase += 16;
        }
        return sum;
    }

    static void dotRowsQ2KPreparedScalar4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
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
        int row0Q = quantsOffset;
        int row1Q = quantsOffset + rowQuantStride;
        int row2Q = quantsOffset + 2 * rowQuantStride;
        int row3Q = quantsOffset + 3 * rowQuantStride;
        int row0Group = matrixGroupBase;
        int row1Group = matrixGroupBase + rowGroupStride;
        int row2Group = matrixGroupBase + 2 * rowGroupStride;
        int row3Group = matrixGroupBase + 3 * rowGroupStride;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            float vectorSum0 = vectorGroupSums[group];
            row0Sum0 += k16MinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group, vectorSum0);
            row1Sum0 += k16MinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group, vectorSum0);
            row2Sum0 += k16MinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group, vectorSum0);
            row3Sum0 += k16MinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group, vectorSum0);

            int vector1 = vBase + 16;
            float vectorSum1 = vectorGroupSums[group + 1];
            row0Sum1 += k16MinBlock(quants, row0Q + 16, vector, vector1, groupScales, groupMins, row0Group + 1, vectorSum1);
            row1Sum1 += k16MinBlock(quants, row1Q + 16, vector, vector1, groupScales, groupMins, row1Group + 1, vectorSum1);
            row2Sum1 += k16MinBlock(quants, row2Q + 16, vector, vector1, groupScales, groupMins, row2Group + 1, vectorSum1);
            row3Sum1 += k16MinBlock(quants, row3Q + 16, vector, vector1, groupScales, groupMins, row3Group + 1, vectorSum1);

            int vector2 = vBase + 32;
            float vectorSum2 = vectorGroupSums[group + 2];
            row0Sum2 += k16MinBlock(quants, row0Q + 32, vector, vector2, groupScales, groupMins, row0Group + 2, vectorSum2);
            row1Sum2 += k16MinBlock(quants, row1Q + 32, vector, vector2, groupScales, groupMins, row1Group + 2, vectorSum2);
            row2Sum2 += k16MinBlock(quants, row2Q + 32, vector, vector2, groupScales, groupMins, row2Group + 2, vectorSum2);
            row3Sum2 += k16MinBlock(quants, row3Q + 32, vector, vector2, groupScales, groupMins, row3Group + 2, vectorSum2);

            int vector3 = vBase + 48;
            float vectorSum3 = vectorGroupSums[group + 3];
            row0Sum3 += k16MinBlock(quants, row0Q + 48, vector, vector3, groupScales, groupMins, row0Group + 3, vectorSum3);
            row1Sum3 += k16MinBlock(quants, row1Q + 48, vector, vector3, groupScales, groupMins, row1Group + 3, vectorSum3);
            row2Sum3 += k16MinBlock(quants, row2Q + 48, vector, vector3, groupScales, groupMins, row2Group + 3, vectorSum3);
            row3Sum3 += k16MinBlock(quants, row3Q + 48, vector, vector3, groupScales, groupMins, row3Group + 3, vectorSum3);

            row0Q += 64;
            row1Q += 64;
            row2Q += 64;
            row3Q += 64;
            row0Group += 4;
            row1Group += 4;
            row2Group += 4;
            row3Group += 4;
            vBase += 64;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; group < groups; group++) {
            float vectorSum = vectorGroupSums[group];
            row0Sum += k16MinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group, vectorSum);
            row1Sum += k16MinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group, vectorSum);
            row2Sum += k16MinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group, vectorSum);
            row3Sum += k16MinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group, vectorSum);
            row0Q += 16;
            row1Q += 16;
            row2Q += 16;
            row3Q += 16;
            row0Group++;
            row1Group++;
            row2Group++;
            row3Group++;
            vBase += 16;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float k16MinBlock(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] groupScales,
            float[] groupMins,
            int matrixGroupBase,
            float vectorSum) {
        return groupScales[matrixGroupBase] * dotSignedByte16Scalar(quants, qBase, vector, vBase)
                - groupMins[matrixGroupBase] * vectorSum;
    }

    static float dotRowQ2KPreparedDirectScalar(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int qBase = quantsOffset;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            sum0 += dotSignedByte16AffineScalar(
                    quants, qBase, vector, vBase, groupScales[matrixGroupBase], -groupMins[matrixGroupBase]);
            sum1 += dotSignedByte16AffineScalar(
                    quants,
                    qBase + 16,
                    vector,
                    vBase + 16,
                    groupScales[matrixGroupBase + 1],
                    -groupMins[matrixGroupBase + 1]);
            sum2 += dotSignedByte16AffineScalar(
                    quants,
                    qBase + 32,
                    vector,
                    vBase + 32,
                    groupScales[matrixGroupBase + 2],
                    -groupMins[matrixGroupBase + 2]);
            sum3 += dotSignedByte16AffineScalar(
                    quants,
                    qBase + 48,
                    vector,
                    vBase + 48,
                    groupScales[matrixGroupBase + 3],
                    -groupMins[matrixGroupBase + 3]);
            matrixGroupBase += 4;
            qBase += 64;
            vBase += 64;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; group < groups; group++) {
            float scale = groupScales[matrixGroupBase];
            float min = groupMins[matrixGroupBase];
            sum += dotSignedByte16AffineScalar(quants, qBase, vector, vBase, scale, -min);
            matrixGroupBase++;
            qBase += 16;
            vBase += 16;
        }
        return sum;
    }

    static void dotRowsQ2KPreparedDirectScalar4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
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
        int row0Q = quantsOffset;
        int row1Q = quantsOffset + rowQuantStride;
        int row2Q = quantsOffset + 2 * rowQuantStride;
        int row3Q = quantsOffset + 3 * rowQuantStride;
        int row0Group = matrixGroupBase;
        int row1Group = matrixGroupBase + rowGroupStride;
        int row2Group = matrixGroupBase + 2 * rowGroupStride;
        int row3Group = matrixGroupBase + 3 * rowGroupStride;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            row0Sum0 += k16DirectMinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group);
            row1Sum0 += k16DirectMinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group);
            row2Sum0 += k16DirectMinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group);
            row3Sum0 += k16DirectMinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group);

            int vector1 = vBase + 16;
            row0Sum1 += k16DirectMinBlock(quants, row0Q + 16, vector, vector1, groupScales, groupMins, row0Group + 1);
            row1Sum1 += k16DirectMinBlock(quants, row1Q + 16, vector, vector1, groupScales, groupMins, row1Group + 1);
            row2Sum1 += k16DirectMinBlock(quants, row2Q + 16, vector, vector1, groupScales, groupMins, row2Group + 1);
            row3Sum1 += k16DirectMinBlock(quants, row3Q + 16, vector, vector1, groupScales, groupMins, row3Group + 1);

            int vector2 = vBase + 32;
            row0Sum2 += k16DirectMinBlock(quants, row0Q + 32, vector, vector2, groupScales, groupMins, row0Group + 2);
            row1Sum2 += k16DirectMinBlock(quants, row1Q + 32, vector, vector2, groupScales, groupMins, row1Group + 2);
            row2Sum2 += k16DirectMinBlock(quants, row2Q + 32, vector, vector2, groupScales, groupMins, row2Group + 2);
            row3Sum2 += k16DirectMinBlock(quants, row3Q + 32, vector, vector2, groupScales, groupMins, row3Group + 2);

            int vector3 = vBase + 48;
            row0Sum3 += k16DirectMinBlock(quants, row0Q + 48, vector, vector3, groupScales, groupMins, row0Group + 3);
            row1Sum3 += k16DirectMinBlock(quants, row1Q + 48, vector, vector3, groupScales, groupMins, row1Group + 3);
            row2Sum3 += k16DirectMinBlock(quants, row2Q + 48, vector, vector3, groupScales, groupMins, row2Group + 3);
            row3Sum3 += k16DirectMinBlock(quants, row3Q + 48, vector, vector3, groupScales, groupMins, row3Group + 3);

            row0Q += 64;
            row1Q += 64;
            row2Q += 64;
            row3Q += 64;
            row0Group += 4;
            row1Group += 4;
            row2Group += 4;
            row3Group += 4;
            vBase += 64;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; group < groups; group++) {
            row0Sum += k16DirectMinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group);
            row1Sum += k16DirectMinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group);
            row2Sum += k16DirectMinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group);
            row3Sum += k16DirectMinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group);
            row0Q += 16;
            row1Q += 16;
            row2Q += 16;
            row3Q += 16;
            row0Group++;
            row1Group++;
            row2Group++;
            row3Group++;
            vBase += 16;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float k16DirectMinBlock(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] groupScales,
            float[] groupMins,
            int matrixGroupBase) {
        return dotSignedByte16AffineScalar(
                quants, qBase, vector, vBase, groupScales[matrixGroupBase], -groupMins[matrixGroupBase]);
    }

    static float dotRowK16PreparedNoMinsVector(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        return dotSignedByte16ScaledBlocksVector(groups, quants, quantsOffset, groupScales, matrixGroupBase, vector);
    }

    static void dotRowsK16PreparedNoMinsVector4(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByte16ScaledBlocksVector4(
                groups,
                quants,
                quantsOffset,
                rowQuantStride,
                groupScales,
                matrixGroupBase,
                rowGroupStride,
                vector,
                output,
                outputOffset);
    }

    static float dotRowK16PreparedNoMinsScalar(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int qBase = quantsOffset;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            sum0 = Math.fma(groupScales[matrixGroupBase], dotSignedByte16Scalar(quants, qBase, vector, vBase), sum0);
            sum1 = Math.fma(
                    groupScales[matrixGroupBase + 1],
                    dotSignedByte16Scalar(quants, qBase + 16, vector, vBase + 16),
                    sum1);
            sum2 = Math.fma(
                    groupScales[matrixGroupBase + 2],
                    dotSignedByte16Scalar(quants, qBase + 32, vector, vBase + 32),
                    sum2);
            sum3 = Math.fma(
                    groupScales[matrixGroupBase + 3],
                    dotSignedByte16Scalar(quants, qBase + 48, vector, vBase + 48),
                    sum3);
            matrixGroupBase += 4;
            qBase += 64;
            vBase += 64;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; group < groups; group++) {
            float quantDot = dotSignedByte16Scalar(quants, qBase, vector, vBase);
            sum = Math.fma(groupScales[matrixGroupBase], quantDot, sum);
            matrixGroupBase++;
            qBase += 16;
            vBase += 16;
        }
        return sum;
    }

    static void dotRowsK16PreparedNoMinsScalar4(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
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
        int row0Q = quantsOffset;
        int row1Q = quantsOffset + rowQuantStride;
        int row2Q = quantsOffset + 2 * rowQuantStride;
        int row3Q = quantsOffset + 3 * rowQuantStride;
        int row0Group = matrixGroupBase;
        int row1Group = matrixGroupBase + rowGroupStride;
        int row2Group = matrixGroupBase + 2 * rowGroupStride;
        int row3Group = matrixGroupBase + 3 * rowGroupStride;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            row0Sum0 = k16NoMinBlock(row0Sum0, quants, row0Q, vector, vBase, groupScales, row0Group);
            row1Sum0 = k16NoMinBlock(row1Sum0, quants, row1Q, vector, vBase, groupScales, row1Group);
            row2Sum0 = k16NoMinBlock(row2Sum0, quants, row2Q, vector, vBase, groupScales, row2Group);
            row3Sum0 = k16NoMinBlock(row3Sum0, quants, row3Q, vector, vBase, groupScales, row3Group);

            int vector1 = vBase + 16;
            row0Sum1 = k16NoMinBlock(row0Sum1, quants, row0Q + 16, vector, vector1, groupScales, row0Group + 1);
            row1Sum1 = k16NoMinBlock(row1Sum1, quants, row1Q + 16, vector, vector1, groupScales, row1Group + 1);
            row2Sum1 = k16NoMinBlock(row2Sum1, quants, row2Q + 16, vector, vector1, groupScales, row2Group + 1);
            row3Sum1 = k16NoMinBlock(row3Sum1, quants, row3Q + 16, vector, vector1, groupScales, row3Group + 1);

            int vector2 = vBase + 32;
            row0Sum2 = k16NoMinBlock(row0Sum2, quants, row0Q + 32, vector, vector2, groupScales, row0Group + 2);
            row1Sum2 = k16NoMinBlock(row1Sum2, quants, row1Q + 32, vector, vector2, groupScales, row1Group + 2);
            row2Sum2 = k16NoMinBlock(row2Sum2, quants, row2Q + 32, vector, vector2, groupScales, row2Group + 2);
            row3Sum2 = k16NoMinBlock(row3Sum2, quants, row3Q + 32, vector, vector2, groupScales, row3Group + 2);

            int vector3 = vBase + 48;
            row0Sum3 = k16NoMinBlock(row0Sum3, quants, row0Q + 48, vector, vector3, groupScales, row0Group + 3);
            row1Sum3 = k16NoMinBlock(row1Sum3, quants, row1Q + 48, vector, vector3, groupScales, row1Group + 3);
            row2Sum3 = k16NoMinBlock(row2Sum3, quants, row2Q + 48, vector, vector3, groupScales, row2Group + 3);
            row3Sum3 = k16NoMinBlock(row3Sum3, quants, row3Q + 48, vector, vector3, groupScales, row3Group + 3);

            row0Q += 64;
            row1Q += 64;
            row2Q += 64;
            row3Q += 64;
            row0Group += 4;
            row1Group += 4;
            row2Group += 4;
            row3Group += 4;
            vBase += 64;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; group < groups; group++) {
            row0Sum = k16NoMinBlock(row0Sum, quants, row0Q, vector, vBase, groupScales, row0Group);
            row1Sum = k16NoMinBlock(row1Sum, quants, row1Q, vector, vBase, groupScales, row1Group);
            row2Sum = k16NoMinBlock(row2Sum, quants, row2Q, vector, vBase, groupScales, row2Group);
            row3Sum = k16NoMinBlock(row3Sum, quants, row3Q, vector, vBase, groupScales, row3Group);
            row0Q += 16;
            row1Q += 16;
            row2Q += 16;
            row3Q += 16;
            row0Group++;
            row1Group++;
            row2Group++;
            row3Group++;
            vBase += 16;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float k16NoMinBlock(
            float sum,
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] groupScales,
            int matrixGroupBase) {
        return Math.fma(groupScales[matrixGroupBase], dotSignedByte16Scalar(quants, qBase, vector, vBase), sum);
    }

    static float dotRowK32GroupsPreparedVector(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector,
            float[] vectorGroupSums) {
        float sum = dotSignedByte32ScaledBlocksVector(groups, quants, quantsOffset, groupScales, matrixGroupBase, vector);
        return sum - dotGroupMinsVector(groups, groupMins, matrixGroupBase, vectorGroupSums);
    }

    static void dotRowsK32GroupsPreparedVector4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int outputOffset) {
        dotSignedByte32ScaledBlocksVector4(
                groups,
                quants,
                quantsOffset,
                rowQuantStride,
                groupScales,
                matrixGroupBase,
                rowGroupStride,
                vector,
                output,
                outputOffset);
        subtractGroupMinsVector4(groups, groupMins, matrixGroupBase, rowGroupStride, vectorGroupSums, output,
                outputOffset);
    }

    static float dotRowK32GroupsPreparedDirectVector(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        return dotSignedByte32AffineBlocksVector(
                groups, quants, quantsOffset, groupScales, groupMins, matrixGroupBase, -1.0f, vector);
    }

    static void dotRowsK32GroupsPreparedDirectVector4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByte32AffineBlocksVector4(
                groups,
                quants,
                quantsOffset,
                rowQuantStride,
                groupScales,
                groupMins,
                matrixGroupBase,
                rowGroupStride,
                -1.0f,
                vector,
                output,
                outputOffset);
    }

    static float dotRowK32GroupsPreparedScalar(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector,
            float[] vectorGroupSums) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int qBase = quantsOffset;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            sum0 += groupScales[matrixGroupBase] * dotSignedByte32Scalar(quants, qBase, vector, vBase)
                    - groupMins[matrixGroupBase] * vectorGroupSums[group];
            sum1 += groupScales[matrixGroupBase + 1]
                            * dotSignedByte32Scalar(quants, qBase + 32, vector, vBase + 32)
                    - groupMins[matrixGroupBase + 1] * vectorGroupSums[group + 1];
            sum2 += groupScales[matrixGroupBase + 2]
                            * dotSignedByte32Scalar(quants, qBase + 64, vector, vBase + 64)
                    - groupMins[matrixGroupBase + 2] * vectorGroupSums[group + 2];
            sum3 += groupScales[matrixGroupBase + 3]
                            * dotSignedByte32Scalar(quants, qBase + 96, vector, vBase + 96)
                    - groupMins[matrixGroupBase + 3] * vectorGroupSums[group + 3];
            matrixGroupBase += 4;
            qBase += 128;
            vBase += 128;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; group < groups; group++) {
            float quantDot = dotSignedByte32Scalar(quants, qBase, vector, vBase);
            float scale = groupScales[matrixGroupBase];
            float min = groupMins[matrixGroupBase];
            float vectorSum = vectorGroupSums[group];
            sum += scale * quantDot - min * vectorSum;
            matrixGroupBase++;
            qBase += 32;
            vBase += 32;
        }
        return sum;
    }

    static void dotRowsK32GroupsPreparedScalar4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
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
        int row0Q = quantsOffset;
        int row1Q = quantsOffset + rowQuantStride;
        int row2Q = quantsOffset + 2 * rowQuantStride;
        int row3Q = quantsOffset + 3 * rowQuantStride;
        int row0Group = matrixGroupBase;
        int row1Group = matrixGroupBase + rowGroupStride;
        int row2Group = matrixGroupBase + 2 * rowGroupStride;
        int row3Group = matrixGroupBase + 3 * rowGroupStride;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            float vectorSum0 = vectorGroupSums[group];
            row0Sum0 += k32MinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group, vectorSum0);
            row1Sum0 += k32MinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group, vectorSum0);
            row2Sum0 += k32MinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group, vectorSum0);
            row3Sum0 += k32MinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group, vectorSum0);

            int vector1 = vBase + 32;
            float vectorSum1 = vectorGroupSums[group + 1];
            row0Sum1 += k32MinBlock(quants, row0Q + 32, vector, vector1, groupScales, groupMins, row0Group + 1, vectorSum1);
            row1Sum1 += k32MinBlock(quants, row1Q + 32, vector, vector1, groupScales, groupMins, row1Group + 1, vectorSum1);
            row2Sum1 += k32MinBlock(quants, row2Q + 32, vector, vector1, groupScales, groupMins, row2Group + 1, vectorSum1);
            row3Sum1 += k32MinBlock(quants, row3Q + 32, vector, vector1, groupScales, groupMins, row3Group + 1, vectorSum1);

            int vector2 = vBase + 64;
            float vectorSum2 = vectorGroupSums[group + 2];
            row0Sum2 += k32MinBlock(quants, row0Q + 64, vector, vector2, groupScales, groupMins, row0Group + 2, vectorSum2);
            row1Sum2 += k32MinBlock(quants, row1Q + 64, vector, vector2, groupScales, groupMins, row1Group + 2, vectorSum2);
            row2Sum2 += k32MinBlock(quants, row2Q + 64, vector, vector2, groupScales, groupMins, row2Group + 2, vectorSum2);
            row3Sum2 += k32MinBlock(quants, row3Q + 64, vector, vector2, groupScales, groupMins, row3Group + 2, vectorSum2);

            int vector3 = vBase + 96;
            float vectorSum3 = vectorGroupSums[group + 3];
            row0Sum3 += k32MinBlock(quants, row0Q + 96, vector, vector3, groupScales, groupMins, row0Group + 3, vectorSum3);
            row1Sum3 += k32MinBlock(quants, row1Q + 96, vector, vector3, groupScales, groupMins, row1Group + 3, vectorSum3);
            row2Sum3 += k32MinBlock(quants, row2Q + 96, vector, vector3, groupScales, groupMins, row2Group + 3, vectorSum3);
            row3Sum3 += k32MinBlock(quants, row3Q + 96, vector, vector3, groupScales, groupMins, row3Group + 3, vectorSum3);

            row0Q += 128;
            row1Q += 128;
            row2Q += 128;
            row3Q += 128;
            row0Group += 4;
            row1Group += 4;
            row2Group += 4;
            row3Group += 4;
            vBase += 128;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; group < groups; group++) {
            float vectorSum = vectorGroupSums[group];
            row0Sum += k32MinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group, vectorSum);
            row1Sum += k32MinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group, vectorSum);
            row2Sum += k32MinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group, vectorSum);
            row3Sum += k32MinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group, vectorSum);
            row0Q += 32;
            row1Q += 32;
            row2Q += 32;
            row3Q += 32;
            row0Group++;
            row1Group++;
            row2Group++;
            row3Group++;
            vBase += 32;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float k32MinBlock(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] groupScales,
            float[] groupMins,
            int matrixGroupBase,
            float vectorSum) {
        return groupScales[matrixGroupBase] * dotSignedByte32Scalar(quants, qBase, vector, vBase)
                - groupMins[matrixGroupBase] * vectorSum;
    }

    static float dotRowK32GroupsPreparedDirectScalar(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int qBase = quantsOffset;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            sum0 += dotSignedByte32AffineScalar(
                    quants, qBase, vector, vBase, groupScales[matrixGroupBase], -groupMins[matrixGroupBase]);
            sum1 += dotSignedByte32AffineScalar(
                    quants,
                    qBase + 32,
                    vector,
                    vBase + 32,
                    groupScales[matrixGroupBase + 1],
                    -groupMins[matrixGroupBase + 1]);
            sum2 += dotSignedByte32AffineScalar(
                    quants,
                    qBase + 64,
                    vector,
                    vBase + 64,
                    groupScales[matrixGroupBase + 2],
                    -groupMins[matrixGroupBase + 2]);
            sum3 += dotSignedByte32AffineScalar(
                    quants,
                    qBase + 96,
                    vector,
                    vBase + 96,
                    groupScales[matrixGroupBase + 3],
                    -groupMins[matrixGroupBase + 3]);
            matrixGroupBase += 4;
            qBase += 128;
            vBase += 128;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; group < groups; group++) {
            float scale = groupScales[matrixGroupBase];
            float min = groupMins[matrixGroupBase];
            sum += dotSignedByte32AffineScalar(quants, qBase, vector, vBase, scale, -min);
            matrixGroupBase++;
            qBase += 32;
            vBase += 32;
        }
        return sum;
    }

    static void dotRowsK32GroupsPreparedDirectScalar4(
            int groups,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
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
        int row0Q = quantsOffset;
        int row1Q = quantsOffset + rowQuantStride;
        int row2Q = quantsOffset + 2 * rowQuantStride;
        int row3Q = quantsOffset + 3 * rowQuantStride;
        int row0Group = matrixGroupBase;
        int row1Group = matrixGroupBase + rowGroupStride;
        int row2Group = matrixGroupBase + 2 * rowGroupStride;
        int row3Group = matrixGroupBase + 3 * rowGroupStride;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            row0Sum0 += k32DirectMinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group);
            row1Sum0 += k32DirectMinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group);
            row2Sum0 += k32DirectMinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group);
            row3Sum0 += k32DirectMinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group);

            int vector1 = vBase + 32;
            row0Sum1 += k32DirectMinBlock(quants, row0Q + 32, vector, vector1, groupScales, groupMins, row0Group + 1);
            row1Sum1 += k32DirectMinBlock(quants, row1Q + 32, vector, vector1, groupScales, groupMins, row1Group + 1);
            row2Sum1 += k32DirectMinBlock(quants, row2Q + 32, vector, vector1, groupScales, groupMins, row2Group + 1);
            row3Sum1 += k32DirectMinBlock(quants, row3Q + 32, vector, vector1, groupScales, groupMins, row3Group + 1);

            int vector2 = vBase + 64;
            row0Sum2 += k32DirectMinBlock(quants, row0Q + 64, vector, vector2, groupScales, groupMins, row0Group + 2);
            row1Sum2 += k32DirectMinBlock(quants, row1Q + 64, vector, vector2, groupScales, groupMins, row1Group + 2);
            row2Sum2 += k32DirectMinBlock(quants, row2Q + 64, vector, vector2, groupScales, groupMins, row2Group + 2);
            row3Sum2 += k32DirectMinBlock(quants, row3Q + 64, vector, vector2, groupScales, groupMins, row3Group + 2);

            int vector3 = vBase + 96;
            row0Sum3 += k32DirectMinBlock(quants, row0Q + 96, vector, vector3, groupScales, groupMins, row0Group + 3);
            row1Sum3 += k32DirectMinBlock(quants, row1Q + 96, vector, vector3, groupScales, groupMins, row1Group + 3);
            row2Sum3 += k32DirectMinBlock(quants, row2Q + 96, vector, vector3, groupScales, groupMins, row2Group + 3);
            row3Sum3 += k32DirectMinBlock(quants, row3Q + 96, vector, vector3, groupScales, groupMins, row3Group + 3);

            row0Q += 128;
            row1Q += 128;
            row2Q += 128;
            row3Q += 128;
            row0Group += 4;
            row1Group += 4;
            row2Group += 4;
            row3Group += 4;
            vBase += 128;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; group < groups; group++) {
            row0Sum += k32DirectMinBlock(quants, row0Q, vector, vBase, groupScales, groupMins, row0Group);
            row1Sum += k32DirectMinBlock(quants, row1Q, vector, vBase, groupScales, groupMins, row1Group);
            row2Sum += k32DirectMinBlock(quants, row2Q, vector, vBase, groupScales, groupMins, row2Group);
            row3Sum += k32DirectMinBlock(quants, row3Q, vector, vBase, groupScales, groupMins, row3Group);
            row0Q += 32;
            row1Q += 32;
            row2Q += 32;
            row3Q += 32;
            row0Group++;
            row1Group++;
            row2Group++;
            row3Group++;
            vBase += 32;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float k32DirectMinBlock(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] groupScales,
            float[] groupMins,
            int matrixGroupBase) {
        return dotSignedByte32AffineScalar(
                quants, qBase, vector, vBase, groupScales[matrixGroupBase], -groupMins[matrixGroupBase]);
    }

    static float dotRowK32GroupsPreparedNoMinsVector(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        return dotSignedByte32ScaledBlocksVector(groups, quants, quantsOffset, groupScales, matrixGroupBase, vector);
    }

    static void dotRowsK32GroupsPreparedNoMinsVector4(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotSignedByte32ScaledBlocksVector4(
                groups,
                quants,
                quantsOffset,
                rowQuantStride,
                groupScales,
                matrixGroupBase,
                rowGroupStride,
                vector,
                output,
                outputOffset);
    }

    static float dotRowK32GroupsPreparedNoMinsScalar(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int matrixGroupBase,
            float[] vector) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int qBase = quantsOffset;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            sum0 = Math.fma(groupScales[matrixGroupBase], dotSignedByte32Scalar(quants, qBase, vector, vBase), sum0);
            sum1 = Math.fma(
                    groupScales[matrixGroupBase + 1],
                    dotSignedByte32Scalar(quants, qBase + 32, vector, vBase + 32),
                    sum1);
            sum2 = Math.fma(
                    groupScales[matrixGroupBase + 2],
                    dotSignedByte32Scalar(quants, qBase + 64, vector, vBase + 64),
                    sum2);
            sum3 = Math.fma(
                    groupScales[matrixGroupBase + 3],
                    dotSignedByte32Scalar(quants, qBase + 96, vector, vBase + 96),
                    sum3);
            matrixGroupBase += 4;
            qBase += 128;
            vBase += 128;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; group < groups; group++) {
            float quantDot = dotSignedByte32Scalar(quants, qBase, vector, vBase);
            sum = Math.fma(groupScales[matrixGroupBase], quantDot, sum);
            matrixGroupBase++;
            qBase += 32;
            vBase += 32;
        }
        return sum;
    }

    static void dotRowsK32GroupsPreparedNoMinsScalar4(
            int groups,
            byte[] quants,
            float[] groupScales,
            int quantsOffset,
            int rowQuantStride,
            int matrixGroupBase,
            int rowGroupStride,
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
        int row0Q = quantsOffset;
        int row1Q = quantsOffset + rowQuantStride;
        int row2Q = quantsOffset + 2 * rowQuantStride;
        int row3Q = quantsOffset + 3 * rowQuantStride;
        int row0Group = matrixGroupBase;
        int row1Group = matrixGroupBase + rowGroupStride;
        int row2Group = matrixGroupBase + 2 * rowGroupStride;
        int row3Group = matrixGroupBase + 3 * rowGroupStride;
        int vBase = 0;
        int group = 0;
        int unrolledLimit = groups - 4;
        for (; group <= unrolledLimit; group += 4) {
            row0Sum0 = k32NoMinBlock(row0Sum0, quants, row0Q, vector, vBase, groupScales, row0Group);
            row1Sum0 = k32NoMinBlock(row1Sum0, quants, row1Q, vector, vBase, groupScales, row1Group);
            row2Sum0 = k32NoMinBlock(row2Sum0, quants, row2Q, vector, vBase, groupScales, row2Group);
            row3Sum0 = k32NoMinBlock(row3Sum0, quants, row3Q, vector, vBase, groupScales, row3Group);

            int vector1 = vBase + 32;
            row0Sum1 = k32NoMinBlock(row0Sum1, quants, row0Q + 32, vector, vector1, groupScales, row0Group + 1);
            row1Sum1 = k32NoMinBlock(row1Sum1, quants, row1Q + 32, vector, vector1, groupScales, row1Group + 1);
            row2Sum1 = k32NoMinBlock(row2Sum1, quants, row2Q + 32, vector, vector1, groupScales, row2Group + 1);
            row3Sum1 = k32NoMinBlock(row3Sum1, quants, row3Q + 32, vector, vector1, groupScales, row3Group + 1);

            int vector2 = vBase + 64;
            row0Sum2 = k32NoMinBlock(row0Sum2, quants, row0Q + 64, vector, vector2, groupScales, row0Group + 2);
            row1Sum2 = k32NoMinBlock(row1Sum2, quants, row1Q + 64, vector, vector2, groupScales, row1Group + 2);
            row2Sum2 = k32NoMinBlock(row2Sum2, quants, row2Q + 64, vector, vector2, groupScales, row2Group + 2);
            row3Sum2 = k32NoMinBlock(row3Sum2, quants, row3Q + 64, vector, vector2, groupScales, row3Group + 2);

            int vector3 = vBase + 96;
            row0Sum3 = k32NoMinBlock(row0Sum3, quants, row0Q + 96, vector, vector3, groupScales, row0Group + 3);
            row1Sum3 = k32NoMinBlock(row1Sum3, quants, row1Q + 96, vector, vector3, groupScales, row1Group + 3);
            row2Sum3 = k32NoMinBlock(row2Sum3, quants, row2Q + 96, vector, vector3, groupScales, row2Group + 3);
            row3Sum3 = k32NoMinBlock(row3Sum3, quants, row3Q + 96, vector, vector3, groupScales, row3Group + 3);

            row0Q += 128;
            row1Q += 128;
            row2Q += 128;
            row3Q += 128;
            row0Group += 4;
            row1Group += 4;
            row2Group += 4;
            row3Group += 4;
            vBase += 128;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; group < groups; group++) {
            row0Sum = k32NoMinBlock(row0Sum, quants, row0Q, vector, vBase, groupScales, row0Group);
            row1Sum = k32NoMinBlock(row1Sum, quants, row1Q, vector, vBase, groupScales, row1Group);
            row2Sum = k32NoMinBlock(row2Sum, quants, row2Q, vector, vBase, groupScales, row2Group);
            row3Sum = k32NoMinBlock(row3Sum, quants, row3Q, vector, vBase, groupScales, row3Group);
            row0Q += 32;
            row1Q += 32;
            row2Q += 32;
            row3Q += 32;
            row0Group++;
            row1Group++;
            row2Group++;
            row3Group++;
            vBase += 32;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float k32NoMinBlock(
            float sum,
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float[] groupScales,
            int matrixGroupBase) {
        return Math.fma(groupScales[matrixGroupBase], dotSignedByte32Scalar(quants, qBase, vector, vBase), sum);
    }

    private static float dotGroupMinsVector(
            int groups,
            float[] groupMins,
            int matrixGroupBase,
            float[] vectorGroupSums) {
        FloatVector acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        int group = 0;
        int unrolledStride = FLOAT_SUM_VECTOR_LANES * 4;
        int unrolledLimit = groups - unrolledStride;
        for (; group <= unrolledLimit; group += unrolledStride) {
            acc0 = FloatVector.fromArray(FLOAT_SUM_SPECIES, groupMins, matrixGroupBase + group)
                    .fma(FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, group), acc0);
            acc1 = FloatVector.fromArray(FLOAT_SUM_SPECIES, groupMins, matrixGroupBase + group + FLOAT_SUM_VECTOR_LANES)
                    .fma(FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, vectorGroupSums, group + FLOAT_SUM_VECTOR_LANES), acc1);
            acc2 = FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, groupMins, matrixGroupBase + group + 2 * FLOAT_SUM_VECTOR_LANES)
                    .fma(FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, vectorGroupSums, group + 2 * FLOAT_SUM_VECTOR_LANES), acc2);
            acc3 = FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, groupMins, matrixGroupBase + group + 3 * FLOAT_SUM_VECTOR_LANES)
                    .fma(FloatVector.fromArray(
                            FLOAT_SUM_SPECIES, vectorGroupSums, group + 3 * FLOAT_SUM_VECTOR_LANES), acc3);
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        int vectorLimit = groups - FLOAT_SUM_VECTOR_LANES;
        for (; group <= vectorLimit; group += FLOAT_SUM_VECTOR_LANES) {
            acc = FloatVector.fromArray(FLOAT_SUM_SPECIES, groupMins, matrixGroupBase + group)
                    .fma(FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, group), acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; group < groups; group++) {
            sum = Math.fma(groupMins[matrixGroupBase + group], vectorGroupSums[group], sum);
        }
        return sum;
    }

    private static void subtractGroupMinsVector4(
            int groups,
            float[] groupMins,
            int matrixGroupBase,
            int rowGroupStride,
            float[] vectorGroupSums,
            float[] output,
            int outputOffset) {
        int row0Base = matrixGroupBase;
        int row1Base = matrixGroupBase + rowGroupStride;
        int row2Base = matrixGroupBase + 2 * rowGroupStride;
        int row3Base = matrixGroupBase + 3 * rowGroupStride;
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
        int group = 0;
        int unrolledStride = FLOAT_SUM_VECTOR_LANES * 4;
        int unrolledLimit = groups - unrolledStride;
        for (; group <= unrolledLimit; group += unrolledStride) {
            FloatVector sums0 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, group);
            row0Acc0 = accumulateGroupMinsVector(groupMins, row0Base + group, sums0, row0Acc0);
            row1Acc0 = accumulateGroupMinsVector(groupMins, row1Base + group, sums0, row1Acc0);
            row2Acc0 = accumulateGroupMinsVector(groupMins, row2Base + group, sums0, row2Acc0);
            row3Acc0 = accumulateGroupMinsVector(groupMins, row3Base + group, sums0, row3Acc0);

            int group1 = group + FLOAT_SUM_VECTOR_LANES;
            FloatVector sums1 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, group1);
            row0Acc1 = accumulateGroupMinsVector(groupMins, row0Base + group1, sums1, row0Acc1);
            row1Acc1 = accumulateGroupMinsVector(groupMins, row1Base + group1, sums1, row1Acc1);
            row2Acc1 = accumulateGroupMinsVector(groupMins, row2Base + group1, sums1, row2Acc1);
            row3Acc1 = accumulateGroupMinsVector(groupMins, row3Base + group1, sums1, row3Acc1);

            int group2 = group + 2 * FLOAT_SUM_VECTOR_LANES;
            FloatVector sums2 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, group2);
            row0Acc2 = accumulateGroupMinsVector(groupMins, row0Base + group2, sums2, row0Acc2);
            row1Acc2 = accumulateGroupMinsVector(groupMins, row1Base + group2, sums2, row1Acc2);
            row2Acc2 = accumulateGroupMinsVector(groupMins, row2Base + group2, sums2, row2Acc2);
            row3Acc2 = accumulateGroupMinsVector(groupMins, row3Base + group2, sums2, row3Acc2);

            int group3 = group + 3 * FLOAT_SUM_VECTOR_LANES;
            FloatVector sums3 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, group3);
            row0Acc3 = accumulateGroupMinsVector(groupMins, row0Base + group3, sums3, row0Acc3);
            row1Acc3 = accumulateGroupMinsVector(groupMins, row1Base + group3, sums3, row1Acc3);
            row2Acc3 = accumulateGroupMinsVector(groupMins, row2Base + group3, sums3, row2Acc3);
            row3Acc3 = accumulateGroupMinsVector(groupMins, row3Base + group3, sums3, row3Acc3);
        }
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        int vectorLimit = groups - FLOAT_SUM_VECTOR_LANES;
        for (; group <= vectorLimit; group += FLOAT_SUM_VECTOR_LANES) {
            FloatVector sums = FloatVector.fromArray(FLOAT_SUM_SPECIES, vectorGroupSums, group);
            row0Acc = accumulateGroupMinsVector(groupMins, row0Base + group, sums, row0Acc);
            row1Acc = accumulateGroupMinsVector(groupMins, row1Base + group, sums, row1Acc);
            row2Acc = accumulateGroupMinsVector(groupMins, row2Base + group, sums, row2Acc);
            row3Acc = accumulateGroupMinsVector(groupMins, row3Base + group, sums, row3Acc);
        }
        float row0Sum = row0Acc.reduceLanes(VectorOperators.ADD);
        float row1Sum = row1Acc.reduceLanes(VectorOperators.ADD);
        float row2Sum = row2Acc.reduceLanes(VectorOperators.ADD);
        float row3Sum = row3Acc.reduceLanes(VectorOperators.ADD);
        for (; group < groups; group++) {
            float vectorSum = vectorGroupSums[group];
            row0Sum = Math.fma(groupMins[row0Base + group], vectorSum, row0Sum);
            row1Sum = Math.fma(groupMins[row1Base + group], vectorSum, row1Sum);
            row2Sum = Math.fma(groupMins[row2Base + group], vectorSum, row2Sum);
            row3Sum = Math.fma(groupMins[row3Base + group], vectorSum, row3Sum);
        }
        output[outputOffset] -= row0Sum;
        output[outputOffset + 1] -= row1Sum;
        output[outputOffset + 2] -= row2Sum;
        output[outputOffset + 3] -= row3Sum;
    }

    private static FloatVector accumulateGroupMinsVector(
            float[] groupMins,
            int groupBase,
            FloatVector vectorGroupSums,
            FloatVector acc) {
        return FloatVector.fromArray(FLOAT_SUM_SPECIES, groupMins, groupBase).fma(vectorGroupSums, acc);
    }
}
