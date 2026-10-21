package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.Q4_DOT_BYTE_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.Q4_DOT_FLOAT_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.Q4_DOT_VECTOR_LANES;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Signed-byte dot kernels for prepared GGUF mat-vec paths.
 *
 * <p>Prepared matrices store unpacked quant values as signed bytes. Keeping the
 * scalar and Vector API reductions here lets {@link GgufTensorOps} focus on row
 * dispatch while this helper owns the low-level arithmetic variants.</p>
 * @author bhangun
 */
final class GgufDot {
    private GgufDot() {
    }

    static float dotSignedByte16Vector(byte[] quants, int qBase, float[] vector, int vBase) {
        FloatVector acc = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        return accumulateSignedByte16Vector(quants, qBase, vector, vBase, acc)
                .reduceLanes(VectorOperators.ADD);
    }

    static float dotSignedByte16ScaledBlocksVector(
            int blocks,
            byte[] quants,
            int qBase,
            float[] scales,
            int scaleBase,
            float[] vector) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            acc0 = accumulateSignedByte16ScaledVector(quants, qBase, vector, vBase, scales[scaleBase], acc0);
            acc1 = accumulateSignedByte16ScaledVector(
                    quants, qBase + 16, vector, vBase + 16, scales[scaleBase + 1], acc1);
            acc2 = accumulateSignedByte16ScaledVector(
                    quants, qBase + 32, vector, vBase + 32, scales[scaleBase + 2], acc2);
            acc3 = accumulateSignedByte16ScaledVector(
                    quants, qBase + 48, vector, vBase + 48, scales[scaleBase + 3], acc3);
            scaleBase += 4;
            qBase += 64;
            vBase += 64;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        for (; block < blocks; block++) {
            acc = accumulateSignedByte16ScaledVector(quants, qBase, vector, vBase, scales[scaleBase], acc);
            scaleBase++;
            qBase += 16;
            vBase += 16;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    static void dotSignedByte16ScaledBlocksVector4(
            int blocks,
            byte[] quants,
            int qBase,
            int qStride,
            float[] scales,
            int scaleBase,
            int scaleStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector row0Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Scale = scaleBase;
        int row1Scale = scaleBase + scaleStride;
        int row2Scale = scaleBase + 2 * scaleStride;
        int row3Scale = scaleBase + 3 * scaleStride;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            row0Acc0 = accumulateSignedByte16ScaledVector(
                    quants, row0Q, vf0, vf1, scales[row0Scale], row0Acc0);
            row1Acc0 = accumulateSignedByte16ScaledVector(
                    quants, row1Q, vf0, vf1, scales[row1Scale], row1Acc0);
            row2Acc0 = accumulateSignedByte16ScaledVector(
                    quants, row2Q, vf0, vf1, scales[row2Scale], row2Acc0);
            row3Acc0 = accumulateSignedByte16ScaledVector(
                    quants, row3Q, vf0, vf1, scales[row3Scale], row3Acc0);

            int vector1 = vBase + 16;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + Q4_DOT_VECTOR_LANES);
            row0Acc1 = accumulateSignedByte16ScaledVector(
                    quants, row0Q + 16, vf0, vf1, scales[row0Scale + 1], row0Acc1);
            row1Acc1 = accumulateSignedByte16ScaledVector(
                    quants, row1Q + 16, vf0, vf1, scales[row1Scale + 1], row1Acc1);
            row2Acc1 = accumulateSignedByte16ScaledVector(
                    quants, row2Q + 16, vf0, vf1, scales[row2Scale + 1], row2Acc1);
            row3Acc1 = accumulateSignedByte16ScaledVector(
                    quants, row3Q + 16, vf0, vf1, scales[row3Scale + 1], row3Acc1);

            int vector2 = vBase + 32;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + Q4_DOT_VECTOR_LANES);
            row0Acc2 = accumulateSignedByte16ScaledVector(
                    quants, row0Q + 32, vf0, vf1, scales[row0Scale + 2], row0Acc2);
            row1Acc2 = accumulateSignedByte16ScaledVector(
                    quants, row1Q + 32, vf0, vf1, scales[row1Scale + 2], row1Acc2);
            row2Acc2 = accumulateSignedByte16ScaledVector(
                    quants, row2Q + 32, vf0, vf1, scales[row2Scale + 2], row2Acc2);
            row3Acc2 = accumulateSignedByte16ScaledVector(
                    quants, row3Q + 32, vf0, vf1, scales[row3Scale + 2], row3Acc2);

            int vector3 = vBase + 48;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + Q4_DOT_VECTOR_LANES);
            row0Acc3 = accumulateSignedByte16ScaledVector(
                    quants, row0Q + 48, vf0, vf1, scales[row0Scale + 3], row0Acc3);
            row1Acc3 = accumulateSignedByte16ScaledVector(
                    quants, row1Q + 48, vf0, vf1, scales[row1Scale + 3], row1Acc3);
            row2Acc3 = accumulateSignedByte16ScaledVector(
                    quants, row2Q + 48, vf0, vf1, scales[row2Scale + 3], row2Acc3);
            row3Acc3 = accumulateSignedByte16ScaledVector(
                    quants, row3Q + 48, vf0, vf1, scales[row3Scale + 3], row3Acc3);

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
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        for (; block < blocks; block++) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            row0Acc = accumulateSignedByte16ScaledVector(
                    quants, row0Q, vf0, vf1, scales[row0Scale], row0Acc);
            row1Acc = accumulateSignedByte16ScaledVector(
                    quants, row1Q, vf0, vf1, scales[row1Scale], row1Acc);
            row2Acc = accumulateSignedByte16ScaledVector(
                    quants, row2Q, vf0, vf1, scales[row2Scale], row2Acc);
            row3Acc = accumulateSignedByte16ScaledVector(
                    quants, row3Q, vf0, vf1, scales[row3Scale], row3Acc);
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
        output[outputOffset] = row0Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 1] = row1Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 2] = row2Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 3] = row3Acc.reduceLanes(VectorOperators.ADD);
    }

    private static FloatVector accumulateSignedByte16Vector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase);
        ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + Q4_DOT_VECTOR_LANES);
        FloatVector qf0 = (FloatVector) q0.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
        acc = qf0.fma(vf0, acc);
        return qf1.fma(vf1, acc);
    }

    private static FloatVector accumulateSignedByte16ScaledVector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            FloatVector acc) {
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
        return accumulateSignedByte16ScaledVector(quants, qBase, vf0, vf1, scale, acc);
    }

    private static FloatVector accumulateSignedByte16ScaledVector(
            byte[] quants,
            int qBase,
            FloatVector vf0,
            FloatVector vf1,
            float scale,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase);
        ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + Q4_DOT_VECTOR_LANES);
        FloatVector scaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scale);
        FloatVector qf0 = (FloatVector) q0.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector blockAcc = qf0.mul(vf0);
        blockAcc = qf1.fma(vf1, blockAcc);
        return blockAcc.fma(scaleVector, acc);
    }

    static float dotSignedByte16AffineVector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            float bias) {
        FloatVector acc = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        return accumulateSignedByte16AffineVector(quants, qBase, vector, vBase, scale, bias, acc)
                .reduceLanes(VectorOperators.ADD);
    }

    static float dotSignedByte16AffineBlocksVector(
            int blocks,
            byte[] quants,
            int qBase,
            float[] scales,
            float[] biases,
            int biasBase,
            float biasScale,
            float[] vector) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            acc0 = accumulateSignedByte16AffineVector(
                    quants, qBase, vector, vBase, scales[biasBase], biases[biasBase] * biasScale, acc0);
            acc1 = accumulateSignedByte16AffineVector(
                    quants,
                    qBase + 16,
                    vector,
                    vBase + 16,
                    scales[biasBase + 1],
                    biases[biasBase + 1] * biasScale,
                    acc1);
            acc2 = accumulateSignedByte16AffineVector(
                    quants,
                    qBase + 32,
                    vector,
                    vBase + 32,
                    scales[biasBase + 2],
                    biases[biasBase + 2] * biasScale,
                    acc2);
            acc3 = accumulateSignedByte16AffineVector(
                    quants,
                    qBase + 48,
                    vector,
                    vBase + 48,
                    scales[biasBase + 3],
                    biases[biasBase + 3] * biasScale,
                    acc3);
            biasBase += 4;
            qBase += 64;
            vBase += 64;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        for (; block < blocks; block++) {
            acc = accumulateSignedByte16AffineVector(
                    quants, qBase, vector, vBase, scales[biasBase], biases[biasBase] * biasScale, acc);
            biasBase++;
            qBase += 16;
            vBase += 16;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    static void dotSignedByte16AffineBlocksVector4(
            int blocks,
            byte[] quants,
            int qBase,
            int qStride,
            float[] scales,
            float[] biases,
            int biasBase,
            int biasStride,
            float biasScale,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector row0Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Bias = biasBase;
        int row1Bias = biasBase + biasStride;
        int row2Bias = biasBase + 2 * biasStride;
        int row3Bias = biasBase + 3 * biasStride;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            row0Acc0 = accumulateSignedByte16AffineVector(
                    quants, row0Q, vf0, vf1, scales[row0Bias], biases[row0Bias] * biasScale, row0Acc0);
            row1Acc0 = accumulateSignedByte16AffineVector(
                    quants, row1Q, vf0, vf1, scales[row1Bias], biases[row1Bias] * biasScale, row1Acc0);
            row2Acc0 = accumulateSignedByte16AffineVector(
                    quants, row2Q, vf0, vf1, scales[row2Bias], biases[row2Bias] * biasScale, row2Acc0);
            row3Acc0 = accumulateSignedByte16AffineVector(
                    quants, row3Q, vf0, vf1, scales[row3Bias], biases[row3Bias] * biasScale, row3Acc0);

            int vector1 = vBase + 16;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + Q4_DOT_VECTOR_LANES);
            row0Acc1 = accumulateSignedByte16AffineVector(
                    quants, row0Q + 16, vf0, vf1, scales[row0Bias + 1], biases[row0Bias + 1] * biasScale, row0Acc1);
            row1Acc1 = accumulateSignedByte16AffineVector(
                    quants, row1Q + 16, vf0, vf1, scales[row1Bias + 1], biases[row1Bias + 1] * biasScale, row1Acc1);
            row2Acc1 = accumulateSignedByte16AffineVector(
                    quants, row2Q + 16, vf0, vf1, scales[row2Bias + 1], biases[row2Bias + 1] * biasScale, row2Acc1);
            row3Acc1 = accumulateSignedByte16AffineVector(
                    quants, row3Q + 16, vf0, vf1, scales[row3Bias + 1], biases[row3Bias + 1] * biasScale, row3Acc1);

            int vector2 = vBase + 32;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + Q4_DOT_VECTOR_LANES);
            row0Acc2 = accumulateSignedByte16AffineVector(
                    quants, row0Q + 32, vf0, vf1, scales[row0Bias + 2], biases[row0Bias + 2] * biasScale, row0Acc2);
            row1Acc2 = accumulateSignedByte16AffineVector(
                    quants, row1Q + 32, vf0, vf1, scales[row1Bias + 2], biases[row1Bias + 2] * biasScale, row1Acc2);
            row2Acc2 = accumulateSignedByte16AffineVector(
                    quants, row2Q + 32, vf0, vf1, scales[row2Bias + 2], biases[row2Bias + 2] * biasScale, row2Acc2);
            row3Acc2 = accumulateSignedByte16AffineVector(
                    quants, row3Q + 32, vf0, vf1, scales[row3Bias + 2], biases[row3Bias + 2] * biasScale, row3Acc2);

            int vector3 = vBase + 48;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + Q4_DOT_VECTOR_LANES);
            row0Acc3 = accumulateSignedByte16AffineVector(
                    quants, row0Q + 48, vf0, vf1, scales[row0Bias + 3], biases[row0Bias + 3] * biasScale, row0Acc3);
            row1Acc3 = accumulateSignedByte16AffineVector(
                    quants, row1Q + 48, vf0, vf1, scales[row1Bias + 3], biases[row1Bias + 3] * biasScale, row1Acc3);
            row2Acc3 = accumulateSignedByte16AffineVector(
                    quants, row2Q + 48, vf0, vf1, scales[row2Bias + 3], biases[row2Bias + 3] * biasScale, row2Acc3);
            row3Acc3 = accumulateSignedByte16AffineVector(
                    quants, row3Q + 48, vf0, vf1, scales[row3Bias + 3], biases[row3Bias + 3] * biasScale, row3Acc3);

            row0Q += 64;
            row1Q += 64;
            row2Q += 64;
            row3Q += 64;
            row0Bias += 4;
            row1Bias += 4;
            row2Bias += 4;
            row3Bias += 4;
            vBase += 64;
        }
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        for (; block < blocks; block++) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            row0Acc = accumulateSignedByte16AffineVector(
                    quants, row0Q, vf0, vf1, scales[row0Bias], biases[row0Bias] * biasScale, row0Acc);
            row1Acc = accumulateSignedByte16AffineVector(
                    quants, row1Q, vf0, vf1, scales[row1Bias], biases[row1Bias] * biasScale, row1Acc);
            row2Acc = accumulateSignedByte16AffineVector(
                    quants, row2Q, vf0, vf1, scales[row2Bias], biases[row2Bias] * biasScale, row2Acc);
            row3Acc = accumulateSignedByte16AffineVector(
                    quants, row3Q, vf0, vf1, scales[row3Bias], biases[row3Bias] * biasScale, row3Acc);
            row0Q += 16;
            row1Q += 16;
            row2Q += 16;
            row3Q += 16;
            row0Bias++;
            row1Bias++;
            row2Bias++;
            row3Bias++;
            vBase += 16;
        }
        output[outputOffset] = row0Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 1] = row1Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 2] = row2Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 3] = row3Acc.reduceLanes(VectorOperators.ADD);
    }

    private static FloatVector accumulateSignedByte16AffineVector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            float bias,
            FloatVector acc) {
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
        return accumulateSignedByte16AffineVector(quants, qBase, vf0, vf1, scale, bias, acc);
    }

    private static FloatVector accumulateSignedByte16AffineVector(
            byte[] quants,
            int qBase,
            FloatVector vf0,
            FloatVector vf1,
            float scale,
            float bias,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase);
        ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + Q4_DOT_VECTOR_LANES);
        FloatVector scaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scale);
        FloatVector biasVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, bias);
        FloatVector qf0 = (FloatVector) q0.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector quantDot = qf0.mul(vf0);
        quantDot = qf1.fma(vf1, quantDot);
        FloatVector vectorSum = vf0.add(vf1);
        acc = quantDot.fma(scaleVector, acc);
        return vectorSum.fma(biasVector, acc);
    }

    static float dotSignedByte16Scalar(byte[] quants, int qBase, float[] vector, int vBase) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (int i = 0; i < 16; i += 4) {
            sum0 += quants[qBase + i] * vector[vBase + i];
            sum1 += quants[qBase + i + 1] * vector[vBase + i + 1];
            sum2 += quants[qBase + i + 2] * vector[vBase + i + 2];
            sum3 += quants[qBase + i + 3] * vector[vBase + i + 3];
        }
        return sum0 + sum1 + sum2 + sum3;
    }

    static float dotSignedByte16AffineScalar(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            float bias) {
        float quantDot0 = 0.0f;
        float quantDot1 = 0.0f;
        float quantDot2 = 0.0f;
        float quantDot3 = 0.0f;
        float vectorSum0 = 0.0f;
        float vectorSum1 = 0.0f;
        float vectorSum2 = 0.0f;
        float vectorSum3 = 0.0f;
        for (int i = 0; i < 16; i += 4) {
            float value0 = vector[vBase + i];
            float value1 = vector[vBase + i + 1];
            float value2 = vector[vBase + i + 2];
            float value3 = vector[vBase + i + 3];
            quantDot0 += quants[qBase + i] * value0;
            quantDot1 += quants[qBase + i + 1] * value1;
            quantDot2 += quants[qBase + i + 2] * value2;
            quantDot3 += quants[qBase + i + 3] * value3;
            vectorSum0 += value0;
            vectorSum1 += value1;
            vectorSum2 += value2;
            vectorSum3 += value3;
        }
        float quantDot = quantDot0 + quantDot1 + quantDot2 + quantDot3;
        float vectorSum = vectorSum0 + vectorSum1 + vectorSum2 + vectorSum3;
        return scale * quantDot + bias * vectorSum;
    }

    static float dotSignedByte32Vector(byte[] quants, int qBase, float[] vector, int vBase) {
        FloatVector acc = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        return accumulateSignedByte32Vector(quants, qBase, vector, vBase, acc)
                .reduceLanes(VectorOperators.ADD);
    }

    static float dotSignedByte32ScaledBlocksVector(
            int blocks,
            byte[] quants,
            int qBase,
            float[] scales,
            int scaleBase,
            float[] vector) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            acc0 = accumulateSignedByte32ScaledVector(quants, qBase, vector, vBase, scales[scaleBase], acc0);
            acc1 = accumulateSignedByte32ScaledVector(
                    quants, qBase + 32, vector, vBase + 32, scales[scaleBase + 1], acc1);
            acc2 = accumulateSignedByte32ScaledVector(
                    quants, qBase + 64, vector, vBase + 64, scales[scaleBase + 2], acc2);
            acc3 = accumulateSignedByte32ScaledVector(
                    quants, qBase + 96, vector, vBase + 96, scales[scaleBase + 3], acc3);
            scaleBase += 4;
            qBase += 128;
            vBase += 128;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        for (; block < blocks; block++) {
            acc = accumulateSignedByte32ScaledVector(quants, qBase, vector, vBase, scales[scaleBase], acc);
            scaleBase++;
            qBase += 32;
            vBase += 32;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    static void dotSignedByte32ScaledBlocksVector4(
            int blocks,
            byte[] quants,
            int qBase,
            int qStride,
            float[] scales,
            int scaleBase,
            int scaleStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector row0Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Scale = scaleBase;
        int row1Scale = scaleBase + scaleStride;
        int row2Scale = scaleBase + 2 * scaleStride;
        int row3Scale = scaleBase + 3 * scaleStride;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc0 = accumulateSignedByte32ScaledVector(
                    quants, row0Q, vf0, vf1, vf2, vf3, scales[row0Scale], row0Acc0);
            row1Acc0 = accumulateSignedByte32ScaledVector(
                    quants, row1Q, vf0, vf1, vf2, vf3, scales[row1Scale], row1Acc0);
            row2Acc0 = accumulateSignedByte32ScaledVector(
                    quants, row2Q, vf0, vf1, vf2, vf3, scales[row2Scale], row2Acc0);
            row3Acc0 = accumulateSignedByte32ScaledVector(
                    quants, row3Q, vf0, vf1, vf2, vf3, scales[row3Scale], row3Acc0);

            int vector1 = vBase + 32;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc1 = accumulateSignedByte32ScaledVector(
                    quants, row0Q + 32, vf0, vf1, vf2, vf3, scales[row0Scale + 1], row0Acc1);
            row1Acc1 = accumulateSignedByte32ScaledVector(
                    quants, row1Q + 32, vf0, vf1, vf2, vf3, scales[row1Scale + 1], row1Acc1);
            row2Acc1 = accumulateSignedByte32ScaledVector(
                    quants, row2Q + 32, vf0, vf1, vf2, vf3, scales[row2Scale + 1], row2Acc1);
            row3Acc1 = accumulateSignedByte32ScaledVector(
                    quants, row3Q + 32, vf0, vf1, vf2, vf3, scales[row3Scale + 1], row3Acc1);

            int vector2 = vBase + 64;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc2 = accumulateSignedByte32ScaledVector(
                    quants, row0Q + 64, vf0, vf1, vf2, vf3, scales[row0Scale + 2], row0Acc2);
            row1Acc2 = accumulateSignedByte32ScaledVector(
                    quants, row1Q + 64, vf0, vf1, vf2, vf3, scales[row1Scale + 2], row1Acc2);
            row2Acc2 = accumulateSignedByte32ScaledVector(
                    quants, row2Q + 64, vf0, vf1, vf2, vf3, scales[row2Scale + 2], row2Acc2);
            row3Acc2 = accumulateSignedByte32ScaledVector(
                    quants, row3Q + 64, vf0, vf1, vf2, vf3, scales[row3Scale + 2], row3Acc2);

            int vector3 = vBase + 96;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc3 = accumulateSignedByte32ScaledVector(
                    quants, row0Q + 96, vf0, vf1, vf2, vf3, scales[row0Scale + 3], row0Acc3);
            row1Acc3 = accumulateSignedByte32ScaledVector(
                    quants, row1Q + 96, vf0, vf1, vf2, vf3, scales[row1Scale + 3], row1Acc3);
            row2Acc3 = accumulateSignedByte32ScaledVector(
                    quants, row2Q + 96, vf0, vf1, vf2, vf3, scales[row2Scale + 3], row2Acc3);
            row3Acc3 = accumulateSignedByte32ScaledVector(
                    quants, row3Q + 96, vf0, vf1, vf2, vf3, scales[row3Scale + 3], row3Acc3);

            row0Q += 128;
            row1Q += 128;
            row2Q += 128;
            row3Q += 128;
            row0Scale += 4;
            row1Scale += 4;
            row2Scale += 4;
            row3Scale += 4;
            vBase += 128;
        }
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        for (; block < blocks; block++) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc = accumulateSignedByte32ScaledVector(
                    quants, row0Q, vf0, vf1, vf2, vf3, scales[row0Scale], row0Acc);
            row1Acc = accumulateSignedByte32ScaledVector(
                    quants, row1Q, vf0, vf1, vf2, vf3, scales[row1Scale], row1Acc);
            row2Acc = accumulateSignedByte32ScaledVector(
                    quants, row2Q, vf0, vf1, vf2, vf3, scales[row2Scale], row2Acc);
            row3Acc = accumulateSignedByte32ScaledVector(
                    quants, row3Q, vf0, vf1, vf2, vf3, scales[row3Scale], row3Acc);
            row0Q += 32;
            row1Q += 32;
            row2Q += 32;
            row3Q += 32;
            row0Scale++;
            row1Scale++;
            row2Scale++;
            row3Scale++;
            vBase += 32;
        }
        output[outputOffset] = row0Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 1] = row1Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 2] = row2Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 3] = row3Acc.reduceLanes(VectorOperators.ADD);
    }

    private static FloatVector accumulateSignedByte32Vector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            FloatVector acc) {
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
        FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 2 * Q4_DOT_VECTOR_LANES);
        FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 3 * Q4_DOT_VECTOR_LANES);
        return accumulateSignedByte32Vector(quants, qBase, vf0, vf1, vf2, vf3, acc);
    }

    private static FloatVector accumulateSignedByteLaneVector(
            byte[] quants,
            int qBase,
            FloatVector vf,
            FloatVector acc) {
        ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase);
        FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        return qf.fma(vf, acc);
    }

    private static FloatVector accumulateSignedByte32Vector(
            byte[] quants,
            int qBase,
            FloatVector vf0,
            FloatVector vf1,
            FloatVector vf2,
            FloatVector vf3,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase);
        ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + Q4_DOT_VECTOR_LANES);
        ByteVector q2 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + 2 * Q4_DOT_VECTOR_LANES);
        ByteVector q3 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + 3 * Q4_DOT_VECTOR_LANES);
        FloatVector qf0 = (FloatVector) q0.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        acc = qf0.fma(vf0, acc);
        acc = qf1.fma(vf1, acc);
        acc = qf2.fma(vf2, acc);
        return qf3.fma(vf3, acc);
    }

    private static FloatVector accumulateSignedByte32ScaledVector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            FloatVector acc) {
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
        FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 2 * Q4_DOT_VECTOR_LANES);
        FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 3 * Q4_DOT_VECTOR_LANES);
        return accumulateSignedByte32ScaledVector(quants, qBase, vf0, vf1, vf2, vf3, scale, acc);
    }

    private static FloatVector accumulateSignedByte32ScaledVector(
            byte[] quants,
            int qBase,
            FloatVector vf0,
            FloatVector vf1,
            FloatVector vf2,
            FloatVector vf3,
            float scale,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase);
        ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + Q4_DOT_VECTOR_LANES);
        ByteVector q2 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + 2 * Q4_DOT_VECTOR_LANES);
        ByteVector q3 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + 3 * Q4_DOT_VECTOR_LANES);
        FloatVector scaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scale);
        FloatVector qf0 = (FloatVector) q0.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector blockAcc = qf0.mul(vf0);
        blockAcc = qf1.fma(vf1, blockAcc);
        blockAcc = qf2.fma(vf2, blockAcc);
        blockAcc = qf3.fma(vf3, blockAcc);
        return blockAcc.fma(scaleVector, acc);
    }

    static float dotSignedByte32AffineVector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            float bias) {
        FloatVector acc = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        return accumulateSignedByte32AffineVector(quants, qBase, vector, vBase, scale, bias, acc)
                .reduceLanes(VectorOperators.ADD);
    }

    static float dotSignedByte32AffineBlocksVector(
            int blocks,
            byte[] quants,
            int qBase,
            float[] scales,
            float[] biases,
            int biasBase,
            float biasScale,
            float[] vector) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            acc0 = accumulateSignedByte32AffineVector(
                    quants, qBase, vector, vBase, scales[biasBase], biases[biasBase] * biasScale, acc0);
            acc1 = accumulateSignedByte32AffineVector(
                    quants,
                    qBase + 32,
                    vector,
                    vBase + 32,
                    scales[biasBase + 1],
                    biases[biasBase + 1] * biasScale,
                    acc1);
            acc2 = accumulateSignedByte32AffineVector(
                    quants,
                    qBase + 64,
                    vector,
                    vBase + 64,
                    scales[biasBase + 2],
                    biases[biasBase + 2] * biasScale,
                    acc2);
            acc3 = accumulateSignedByte32AffineVector(
                    quants,
                    qBase + 96,
                    vector,
                    vBase + 96,
                    scales[biasBase + 3],
                    biases[biasBase + 3] * biasScale,
                    acc3);
            biasBase += 4;
            qBase += 128;
            vBase += 128;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        for (; block < blocks; block++) {
            acc = accumulateSignedByte32AffineVector(
                    quants, qBase, vector, vBase, scales[biasBase], biases[biasBase] * biasScale, acc);
            biasBase++;
            qBase += 32;
            vBase += 32;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    static void dotSignedByte32AffineBlocksVector4(
            int blocks,
            byte[] quants,
            int qBase,
            int qStride,
            float[] scales,
            float[] biases,
            int biasBase,
            int biasStride,
            float biasScale,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector row0Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Bias = biasBase;
        int row1Bias = biasBase + biasStride;
        int row2Bias = biasBase + 2 * biasStride;
        int row3Bias = biasBase + 3 * biasStride;
        int vBase = 0;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc0 = accumulateSignedByte32AffineVector(
                    quants, row0Q, vf0, vf1, vf2, vf3, scales[row0Bias], biases[row0Bias] * biasScale, row0Acc0);
            row1Acc0 = accumulateSignedByte32AffineVector(
                    quants, row1Q, vf0, vf1, vf2, vf3, scales[row1Bias], biases[row1Bias] * biasScale, row1Acc0);
            row2Acc0 = accumulateSignedByte32AffineVector(
                    quants, row2Q, vf0, vf1, vf2, vf3, scales[row2Bias], biases[row2Bias] * biasScale, row2Acc0);
            row3Acc0 = accumulateSignedByte32AffineVector(
                    quants, row3Q, vf0, vf1, vf2, vf3, scales[row3Bias], biases[row3Bias] * biasScale, row3Acc0);

            int vector1 = vBase + 32;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc1 = accumulateSignedByte32AffineVector(
                    quants, row0Q + 32, vf0, vf1, vf2, vf3, scales[row0Bias + 1],
                    biases[row0Bias + 1] * biasScale, row0Acc1);
            row1Acc1 = accumulateSignedByte32AffineVector(
                    quants, row1Q + 32, vf0, vf1, vf2, vf3, scales[row1Bias + 1],
                    biases[row1Bias + 1] * biasScale, row1Acc1);
            row2Acc1 = accumulateSignedByte32AffineVector(
                    quants, row2Q + 32, vf0, vf1, vf2, vf3, scales[row2Bias + 1],
                    biases[row2Bias + 1] * biasScale, row2Acc1);
            row3Acc1 = accumulateSignedByte32AffineVector(
                    quants, row3Q + 32, vf0, vf1, vf2, vf3, scales[row3Bias + 1],
                    biases[row3Bias + 1] * biasScale, row3Acc1);

            int vector2 = vBase + 64;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc2 = accumulateSignedByte32AffineVector(
                    quants, row0Q + 64, vf0, vf1, vf2, vf3, scales[row0Bias + 2],
                    biases[row0Bias + 2] * biasScale, row0Acc2);
            row1Acc2 = accumulateSignedByte32AffineVector(
                    quants, row1Q + 64, vf0, vf1, vf2, vf3, scales[row1Bias + 2],
                    biases[row1Bias + 2] * biasScale, row1Acc2);
            row2Acc2 = accumulateSignedByte32AffineVector(
                    quants, row2Q + 64, vf0, vf1, vf2, vf3, scales[row2Bias + 2],
                    biases[row2Bias + 2] * biasScale, row2Acc2);
            row3Acc2 = accumulateSignedByte32AffineVector(
                    quants, row3Q + 64, vf0, vf1, vf2, vf3, scales[row3Bias + 2],
                    biases[row3Bias + 2] * biasScale, row3Acc2);

            int vector3 = vBase + 96;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc3 = accumulateSignedByte32AffineVector(
                    quants, row0Q + 96, vf0, vf1, vf2, vf3, scales[row0Bias + 3],
                    biases[row0Bias + 3] * biasScale, row0Acc3);
            row1Acc3 = accumulateSignedByte32AffineVector(
                    quants, row1Q + 96, vf0, vf1, vf2, vf3, scales[row1Bias + 3],
                    biases[row1Bias + 3] * biasScale, row1Acc3);
            row2Acc3 = accumulateSignedByte32AffineVector(
                    quants, row2Q + 96, vf0, vf1, vf2, vf3, scales[row2Bias + 3],
                    biases[row2Bias + 3] * biasScale, row2Acc3);
            row3Acc3 = accumulateSignedByte32AffineVector(
                    quants, row3Q + 96, vf0, vf1, vf2, vf3, scales[row3Bias + 3],
                    biases[row3Bias + 3] * biasScale, row3Acc3);

            row0Q += 128;
            row1Q += 128;
            row2Q += 128;
            row3Q += 128;
            row0Bias += 4;
            row1Bias += 4;
            row2Bias += 4;
            row3Bias += 4;
            vBase += 128;
        }
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        for (; block < blocks; block++) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 3 * Q4_DOT_VECTOR_LANES);
            row0Acc = accumulateSignedByte32AffineVector(
                    quants, row0Q, vf0, vf1, vf2, vf3, scales[row0Bias], biases[row0Bias] * biasScale, row0Acc);
            row1Acc = accumulateSignedByte32AffineVector(
                    quants, row1Q, vf0, vf1, vf2, vf3, scales[row1Bias], biases[row1Bias] * biasScale, row1Acc);
            row2Acc = accumulateSignedByte32AffineVector(
                    quants, row2Q, vf0, vf1, vf2, vf3, scales[row2Bias], biases[row2Bias] * biasScale, row2Acc);
            row3Acc = accumulateSignedByte32AffineVector(
                    quants, row3Q, vf0, vf1, vf2, vf3, scales[row3Bias], biases[row3Bias] * biasScale, row3Acc);
            row0Q += 32;
            row1Q += 32;
            row2Q += 32;
            row3Q += 32;
            row0Bias++;
            row1Bias++;
            row2Bias++;
            row3Bias++;
            vBase += 32;
        }
        output[outputOffset] = row0Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 1] = row1Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 2] = row2Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 3] = row3Acc.reduceLanes(VectorOperators.ADD);
    }

    private static FloatVector accumulateSignedByte32AffineVector(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            float bias,
            FloatVector acc) {
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + Q4_DOT_VECTOR_LANES);
        FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 2 * Q4_DOT_VECTOR_LANES);
        FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + 3 * Q4_DOT_VECTOR_LANES);
        return accumulateSignedByte32AffineVector(quants, qBase, vf0, vf1, vf2, vf3, scale, bias, acc);
    }

    private static FloatVector accumulateSignedByte32AffineVector(
            byte[] quants,
            int qBase,
            FloatVector vf0,
            FloatVector vf1,
            FloatVector vf2,
            FloatVector vf3,
            float scale,
            float bias,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase);
        ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + Q4_DOT_VECTOR_LANES);
        ByteVector q2 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + 2 * Q4_DOT_VECTOR_LANES);
        ByteVector q3 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + 3 * Q4_DOT_VECTOR_LANES);
        FloatVector scaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scale);
        FloatVector biasVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, bias);
        FloatVector qf0 = (FloatVector) q0.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector quantDot = qf0.mul(vf0);
        quantDot = qf1.fma(vf1, quantDot);
        quantDot = qf2.fma(vf2, quantDot);
        quantDot = qf3.fma(vf3, quantDot);
        FloatVector vectorSum = vf0.add(vf1).add(vf2).add(vf3);
        acc = quantDot.fma(scaleVector, acc);
        return vectorSum.fma(biasVector, acc);
    }

    static float dotSignedByte32Scalar(byte[] quants, int qBase, float[] vector, int vBase) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (int i = 0; i < 32; i += 4) {
            sum0 += quants[qBase + i] * vector[vBase + i];
            sum1 += quants[qBase + i + 1] * vector[vBase + i + 1];
            sum2 += quants[qBase + i + 2] * vector[vBase + i + 2];
            sum3 += quants[qBase + i + 3] * vector[vBase + i + 3];
        }
        return sum0 + sum1 + sum2 + sum3;
    }

    static float dotSignedByte32AffineScalar(
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            float bias) {
        float quantDot0 = 0.0f;
        float quantDot1 = 0.0f;
        float quantDot2 = 0.0f;
        float quantDot3 = 0.0f;
        float vectorSum0 = 0.0f;
        float vectorSum1 = 0.0f;
        float vectorSum2 = 0.0f;
        float vectorSum3 = 0.0f;
        for (int i = 0; i < 32; i += 4) {
            float value0 = vector[vBase + i];
            float value1 = vector[vBase + i + 1];
            float value2 = vector[vBase + i + 2];
            float value3 = vector[vBase + i + 3];
            quantDot0 += quants[qBase + i] * value0;
            quantDot1 += quants[qBase + i + 1] * value1;
            quantDot2 += quants[qBase + i + 2] * value2;
            quantDot3 += quants[qBase + i + 3] * value3;
            vectorSum0 += value0;
            vectorSum1 += value1;
            vectorSum2 += value2;
            vectorSum3 += value3;
        }
        float quantDot = quantDot0 + quantDot1 + quantDot2 + quantDot3;
        float vectorSum = vectorSum0 + vectorSum1 + vectorSum2 + vectorSum3;
        return scale * quantDot + bias * vectorSum;
    }

    static float dotSignedByteWideBlockVector(byte[] quants, int qBase, float[] vector, int vBase, int length) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int unrolledStride = Q4_DOT_VECTOR_LANES * 4;
        for (int i = 0; i < length; i += unrolledStride) {
            ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
            ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + Q4_DOT_VECTOR_LANES);
            ByteVector q2 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 2 * Q4_DOT_VECTOR_LANES);
            ByteVector q3 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 3 * Q4_DOT_VECTOR_LANES);
            FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 3 * Q4_DOT_VECTOR_LANES);
            acc0 = qf.fma(vf, acc0);
            acc1 = qf1.fma(vf1, acc1);
            acc2 = qf2.fma(vf2, acc2);
            acc3 = qf3.fma(vf3, acc3);
        }
        return acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
    }

    static float dotSignedByteWideScaledBlocksVector(
            int blocks,
            int blockSize,
            byte[] quants,
            int qBase,
            float[] scales,
            int scaleBase,
            float[] vector) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int vBase = 0;
        int unrolledStride = Q4_DOT_VECTOR_LANES * 4;
        for (int block = 0; block < blocks; block++) {
            FloatVector scale = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scales[scaleBase]);
            FloatVector blockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector blockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector blockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector blockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            for (int i = 0; i < blockSize; i += unrolledStride) {
                ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
                ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + Q4_DOT_VECTOR_LANES);
                ByteVector q2 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 2 * Q4_DOT_VECTOR_LANES);
                ByteVector q3 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 3 * Q4_DOT_VECTOR_LANES);
                FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
                FloatVector vf1 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + Q4_DOT_VECTOR_LANES);
                FloatVector vf2 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 2 * Q4_DOT_VECTOR_LANES);
                FloatVector vf3 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 3 * Q4_DOT_VECTOR_LANES);
                blockAcc0 = qf.fma(vf, blockAcc0);
                blockAcc1 = qf1.fma(vf1, blockAcc1);
                blockAcc2 = qf2.fma(vf2, blockAcc2);
                blockAcc3 = qf3.fma(vf3, blockAcc3);
            }
            acc0 = blockAcc0.fma(scale, acc0);
            acc1 = blockAcc1.fma(scale, acc1);
            acc2 = blockAcc2.fma(scale, acc2);
            acc3 = blockAcc3.fma(scale, acc3);
            scaleBase++;
            qBase += blockSize;
            vBase += blockSize;
        }
        return acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
    }

    static void dotSignedByteWideScaledBlocksVector4(
            int blocks,
            int blockSize,
            byte[] quants,
            int qBase,
            int qStride,
            float[] scales,
            int scaleBase,
            int scaleStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector row0Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Scale = scaleBase;
        int row1Scale = scaleBase + scaleStride;
        int row2Scale = scaleBase + 2 * scaleStride;
        int row3Scale = scaleBase + 3 * scaleStride;
        int vBase = 0;
        int unrolledStride = Q4_DOT_VECTOR_LANES * 4;
        for (int block = 0; block < blocks; block++) {
            FloatVector row0BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row0BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row0BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row0BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            for (int i = 0; i < blockSize; i += unrolledStride) {
                FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
                FloatVector vf1 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + Q4_DOT_VECTOR_LANES);
                FloatVector vf2 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 2 * Q4_DOT_VECTOR_LANES);
                FloatVector vf3 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 3 * Q4_DOT_VECTOR_LANES);
                row0BlockAcc0 = accumulateSignedByteLaneVector(quants, row0Q + i, vf0, row0BlockAcc0);
                row0BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row0Q + i + Q4_DOT_VECTOR_LANES, vf1, row0BlockAcc1);
                row0BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row0Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row0BlockAcc2);
                row0BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row0Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row0BlockAcc3);
                row1BlockAcc0 = accumulateSignedByteLaneVector(quants, row1Q + i, vf0, row1BlockAcc0);
                row1BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row1Q + i + Q4_DOT_VECTOR_LANES, vf1, row1BlockAcc1);
                row1BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row1Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row1BlockAcc2);
                row1BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row1Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row1BlockAcc3);
                row2BlockAcc0 = accumulateSignedByteLaneVector(quants, row2Q + i, vf0, row2BlockAcc0);
                row2BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row2Q + i + Q4_DOT_VECTOR_LANES, vf1, row2BlockAcc1);
                row2BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row2Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row2BlockAcc2);
                row2BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row2Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row2BlockAcc3);
                row3BlockAcc0 = accumulateSignedByteLaneVector(quants, row3Q + i, vf0, row3BlockAcc0);
                row3BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row3Q + i + Q4_DOT_VECTOR_LANES, vf1, row3BlockAcc1);
                row3BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row3Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row3BlockAcc2);
                row3BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row3Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row3BlockAcc3);
            }
            FloatVector row0ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scales[row0Scale]);
            FloatVector row1ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scales[row1Scale]);
            FloatVector row2ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scales[row2Scale]);
            FloatVector row3ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scales[row3Scale]);
            row0Acc0 = row0BlockAcc0.fma(row0ScaleVector, row0Acc0);
            row0Acc1 = row0BlockAcc1.fma(row0ScaleVector, row0Acc1);
            row0Acc2 = row0BlockAcc2.fma(row0ScaleVector, row0Acc2);
            row0Acc3 = row0BlockAcc3.fma(row0ScaleVector, row0Acc3);
            row1Acc0 = row1BlockAcc0.fma(row1ScaleVector, row1Acc0);
            row1Acc1 = row1BlockAcc1.fma(row1ScaleVector, row1Acc1);
            row1Acc2 = row1BlockAcc2.fma(row1ScaleVector, row1Acc2);
            row1Acc3 = row1BlockAcc3.fma(row1ScaleVector, row1Acc3);
            row2Acc0 = row2BlockAcc0.fma(row2ScaleVector, row2Acc0);
            row2Acc1 = row2BlockAcc1.fma(row2ScaleVector, row2Acc1);
            row2Acc2 = row2BlockAcc2.fma(row2ScaleVector, row2Acc2);
            row2Acc3 = row2BlockAcc3.fma(row2ScaleVector, row2Acc3);
            row3Acc0 = row3BlockAcc0.fma(row3ScaleVector, row3Acc0);
            row3Acc1 = row3BlockAcc1.fma(row3ScaleVector, row3Acc1);
            row3Acc2 = row3BlockAcc2.fma(row3ScaleVector, row3Acc2);
            row3Acc3 = row3BlockAcc3.fma(row3ScaleVector, row3Acc3);
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
        output[outputOffset] = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3).reduceLanes(VectorOperators.ADD);
        output[outputOffset + 1] = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3).reduceLanes(VectorOperators.ADD);
        output[outputOffset + 2] = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3).reduceLanes(VectorOperators.ADD);
        output[outputOffset + 3] = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3).reduceLanes(VectorOperators.ADD);
    }

    static float dotSignedByteWideBlockScalar(byte[] quants, int qBase, float[] vector, int vBase, int length) {
        return dotSignedByteBlockScalar(quants, qBase, vector, vBase, length);
    }

    static float dotSignedByteBlockVector(byte[] quants, int qBase, float[] vector, int vBase, int length) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int i = 0;
        int unrolledStride = Q4_DOT_VECTOR_LANES * 4;
        int unrolledLimit = length - unrolledStride;
        for (; i <= unrolledLimit; i += unrolledStride) {
            ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
            ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + Q4_DOT_VECTOR_LANES);
            ByteVector q2 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 2 * Q4_DOT_VECTOR_LANES);
            ByteVector q3 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 3 * Q4_DOT_VECTOR_LANES);
            FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
            FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 3 * Q4_DOT_VECTOR_LANES);
            acc0 = qf.fma(vf, acc0);
            acc1 = qf1.fma(vf1, acc1);
            acc2 = qf2.fma(vf2, acc2);
            acc3 = qf3.fma(vf3, acc3);
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        int vectorLimit = length - Q4_DOT_VECTOR_LANES;
        for (; i <= vectorLimit; i += Q4_DOT_VECTOR_LANES) {
            ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
            FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
            acc = qf.fma(vf, acc);
        }
        float quantDot = acc.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            quantDot += quants[qBase + i] * vector[vBase + i];
        }
        return quantDot;
    }

    static float dotSignedByteScaledBlocksVector(
            int blocks,
            int blockSize,
            byte[] quants,
            int qBase,
            float[] scales,
            int scaleBase,
            float[] vector) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        float scalarTail = 0.0f;
        int vBase = 0;
        int unrolledStride = Q4_DOT_VECTOR_LANES * 4;
        for (int block = 0; block < blocks; block++) {
            FloatVector scaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scales[scaleBase]);
            FloatVector blockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector blockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector blockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector blockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            float blockTail = 0.0f;
            int i = 0;
            int unrolledLimit = blockSize - unrolledStride;
            for (; i <= unrolledLimit; i += unrolledStride) {
                ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
                ByteVector q1 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + Q4_DOT_VECTOR_LANES);
                ByteVector q2 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 2 * Q4_DOT_VECTOR_LANES);
                ByteVector q3 = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i + 3 * Q4_DOT_VECTOR_LANES);
                FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
                FloatVector vf1 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + Q4_DOT_VECTOR_LANES);
                FloatVector vf2 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 2 * Q4_DOT_VECTOR_LANES);
                FloatVector vf3 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 3 * Q4_DOT_VECTOR_LANES);
                blockAcc0 = qf.fma(vf, blockAcc0);
                blockAcc1 = qf1.fma(vf1, blockAcc1);
                blockAcc2 = qf2.fma(vf2, blockAcc2);
                blockAcc3 = qf3.fma(vf3, blockAcc3);
            }
            int vectorLimit = blockSize - Q4_DOT_VECTOR_LANES;
            for (; i <= vectorLimit; i += Q4_DOT_VECTOR_LANES) {
                ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
                FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
                FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
                blockAcc0 = qf.fma(vf, blockAcc0);
            }
            float scale = scales[scaleBase];
            for (; i < blockSize; i++) {
                blockTail = Math.fma(quants[qBase + i], vector[vBase + i], blockTail);
            }
            acc0 = blockAcc0.fma(scaleVector, acc0);
            acc1 = blockAcc1.fma(scaleVector, acc1);
            acc2 = blockAcc2.fma(scaleVector, acc2);
            acc3 = blockAcc3.fma(scaleVector, acc3);
            scalarTail = Math.fma(scale, blockTail, scalarTail);
            scaleBase++;
            qBase += blockSize;
            vBase += blockSize;
        }
        return acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD) + scalarTail;
    }

    static void dotSignedByteScaledBlocksVector4(
            int blocks,
            int blockSize,
            byte[] quants,
            int qBase,
            int qStride,
            float[] scales,
            int scaleBase,
            int scaleStride,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector row0Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row0Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row1Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row2Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector row3Acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        float row0ScalarTail = 0.0f;
        float row1ScalarTail = 0.0f;
        float row2ScalarTail = 0.0f;
        float row3ScalarTail = 0.0f;
        int row0Q = qBase;
        int row1Q = qBase + qStride;
        int row2Q = qBase + 2 * qStride;
        int row3Q = qBase + 3 * qStride;
        int row0Scale = scaleBase;
        int row1Scale = scaleBase + scaleStride;
        int row2Scale = scaleBase + 2 * scaleStride;
        int row3Scale = scaleBase + 3 * scaleStride;
        int vBase = 0;
        int unrolledStride = Q4_DOT_VECTOR_LANES * 4;
        for (int block = 0; block < blocks; block++) {
            FloatVector row0BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row0BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row0BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row0BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row1BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row2BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            FloatVector row3BlockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
            float row0BlockTail = 0.0f;
            float row1BlockTail = 0.0f;
            float row2BlockTail = 0.0f;
            float row3BlockTail = 0.0f;
            int i = 0;
            int unrolledLimit = blockSize - unrolledStride;
            for (; i <= unrolledLimit; i += unrolledStride) {
                FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
                FloatVector vf1 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + Q4_DOT_VECTOR_LANES);
                FloatVector vf2 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 2 * Q4_DOT_VECTOR_LANES);
                FloatVector vf3 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, vBase + i + 3 * Q4_DOT_VECTOR_LANES);
                row0BlockAcc0 = accumulateSignedByteLaneVector(quants, row0Q + i, vf0, row0BlockAcc0);
                row0BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row0Q + i + Q4_DOT_VECTOR_LANES, vf1, row0BlockAcc1);
                row0BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row0Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row0BlockAcc2);
                row0BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row0Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row0BlockAcc3);
                row1BlockAcc0 = accumulateSignedByteLaneVector(quants, row1Q + i, vf0, row1BlockAcc0);
                row1BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row1Q + i + Q4_DOT_VECTOR_LANES, vf1, row1BlockAcc1);
                row1BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row1Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row1BlockAcc2);
                row1BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row1Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row1BlockAcc3);
                row2BlockAcc0 = accumulateSignedByteLaneVector(quants, row2Q + i, vf0, row2BlockAcc0);
                row2BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row2Q + i + Q4_DOT_VECTOR_LANES, vf1, row2BlockAcc1);
                row2BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row2Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row2BlockAcc2);
                row2BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row2Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row2BlockAcc3);
                row3BlockAcc0 = accumulateSignedByteLaneVector(quants, row3Q + i, vf0, row3BlockAcc0);
                row3BlockAcc1 = accumulateSignedByteLaneVector(
                        quants, row3Q + i + Q4_DOT_VECTOR_LANES, vf1, row3BlockAcc1);
                row3BlockAcc2 = accumulateSignedByteLaneVector(
                        quants, row3Q + i + 2 * Q4_DOT_VECTOR_LANES, vf2, row3BlockAcc2);
                row3BlockAcc3 = accumulateSignedByteLaneVector(
                        quants, row3Q + i + 3 * Q4_DOT_VECTOR_LANES, vf3, row3BlockAcc3);
            }
            int vectorLimit = blockSize - Q4_DOT_VECTOR_LANES;
            for (; i <= vectorLimit; i += Q4_DOT_VECTOR_LANES) {
                FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
                row0BlockAcc0 = accumulateSignedByteLaneVector(quants, row0Q + i, vf, row0BlockAcc0);
                row1BlockAcc0 = accumulateSignedByteLaneVector(quants, row1Q + i, vf, row1BlockAcc0);
                row2BlockAcc0 = accumulateSignedByteLaneVector(quants, row2Q + i, vf, row2BlockAcc0);
                row3BlockAcc0 = accumulateSignedByteLaneVector(quants, row3Q + i, vf, row3BlockAcc0);
            }
            for (; i < blockSize; i++) {
                float value = vector[vBase + i];
                row0BlockTail = Math.fma(quants[row0Q + i], value, row0BlockTail);
                row1BlockTail = Math.fma(quants[row1Q + i], value, row1BlockTail);
                row2BlockTail = Math.fma(quants[row2Q + i], value, row2BlockTail);
                row3BlockTail = Math.fma(quants[row3Q + i], value, row3BlockTail);
            }
            float row0ScaleValue = scales[row0Scale];
            float row1ScaleValue = scales[row1Scale];
            float row2ScaleValue = scales[row2Scale];
            float row3ScaleValue = scales[row3Scale];
            FloatVector row0ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, row0ScaleValue);
            FloatVector row1ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, row1ScaleValue);
            FloatVector row2ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, row2ScaleValue);
            FloatVector row3ScaleVector = FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, row3ScaleValue);
            row0Acc0 = row0BlockAcc0.fma(row0ScaleVector, row0Acc0);
            row0Acc1 = row0BlockAcc1.fma(row0ScaleVector, row0Acc1);
            row0Acc2 = row0BlockAcc2.fma(row0ScaleVector, row0Acc2);
            row0Acc3 = row0BlockAcc3.fma(row0ScaleVector, row0Acc3);
            row1Acc0 = row1BlockAcc0.fma(row1ScaleVector, row1Acc0);
            row1Acc1 = row1BlockAcc1.fma(row1ScaleVector, row1Acc1);
            row1Acc2 = row1BlockAcc2.fma(row1ScaleVector, row1Acc2);
            row1Acc3 = row1BlockAcc3.fma(row1ScaleVector, row1Acc3);
            row2Acc0 = row2BlockAcc0.fma(row2ScaleVector, row2Acc0);
            row2Acc1 = row2BlockAcc1.fma(row2ScaleVector, row2Acc1);
            row2Acc2 = row2BlockAcc2.fma(row2ScaleVector, row2Acc2);
            row2Acc3 = row2BlockAcc3.fma(row2ScaleVector, row2Acc3);
            row3Acc0 = row3BlockAcc0.fma(row3ScaleVector, row3Acc0);
            row3Acc1 = row3BlockAcc1.fma(row3ScaleVector, row3Acc1);
            row3Acc2 = row3BlockAcc2.fma(row3ScaleVector, row3Acc2);
            row3Acc3 = row3BlockAcc3.fma(row3ScaleVector, row3Acc3);
            row0ScalarTail = Math.fma(row0ScaleValue, row0BlockTail, row0ScalarTail);
            row1ScalarTail = Math.fma(row1ScaleValue, row1BlockTail, row1ScalarTail);
            row2ScalarTail = Math.fma(row2ScaleValue, row2BlockTail, row2ScalarTail);
            row3ScalarTail = Math.fma(row3ScaleValue, row3BlockTail, row3ScalarTail);
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
        output[outputOffset] =
                row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3).reduceLanes(VectorOperators.ADD) + row0ScalarTail;
        output[outputOffset + 1] =
                row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3).reduceLanes(VectorOperators.ADD) + row1ScalarTail;
        output[outputOffset + 2] =
                row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3).reduceLanes(VectorOperators.ADD) + row2ScalarTail;
        output[outputOffset + 3] =
                row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3).reduceLanes(VectorOperators.ADD) + row3ScalarTail;
    }

    static float dotSignedByteBlockScalar(byte[] quants, int qBase, float[] vector, int vBase, int length) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int i = 0;
        int unrolledLimit = length - 16;
        for (; i <= unrolledLimit; i += 16) {
            sum0 += quants[qBase + i] * vector[vBase + i];
            sum1 += quants[qBase + i + 1] * vector[vBase + i + 1];
            sum2 += quants[qBase + i + 2] * vector[vBase + i + 2];
            sum3 += quants[qBase + i + 3] * vector[vBase + i + 3];
            sum0 += quants[qBase + i + 4] * vector[vBase + i + 4];
            sum1 += quants[qBase + i + 5] * vector[vBase + i + 5];
            sum2 += quants[qBase + i + 6] * vector[vBase + i + 6];
            sum3 += quants[qBase + i + 7] * vector[vBase + i + 7];
            sum0 += quants[qBase + i + 8] * vector[vBase + i + 8];
            sum1 += quants[qBase + i + 9] * vector[vBase + i + 9];
            sum2 += quants[qBase + i + 10] * vector[vBase + i + 10];
            sum3 += quants[qBase + i + 11] * vector[vBase + i + 11];
            sum0 += quants[qBase + i + 12] * vector[vBase + i + 12];
            sum1 += quants[qBase + i + 13] * vector[vBase + i + 13];
            sum2 += quants[qBase + i + 14] * vector[vBase + i + 14];
            sum3 += quants[qBase + i + 15] * vector[vBase + i + 15];
        }
        int pairLimit = length - 4;
        for (; i <= pairLimit; i += 4) {
            sum0 += quants[qBase + i] * vector[vBase + i];
            sum1 += quants[qBase + i + 1] * vector[vBase + i + 1];
            sum2 += quants[qBase + i + 2] * vector[vBase + i + 2];
            sum3 += quants[qBase + i + 3] * vector[vBase + i + 3];
        }
        float quantDot = sum0 + sum1 + sum2 + sum3;
        for (; i < length; i++) {
            quantDot += quants[qBase + i] * vector[vBase + i];
        }
        return quantDot;
    }

}
