package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_VECTOR_LANES;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KWorkBuffer;

/**
 * Vector-group sum helpers for affine prepared GGUF mat-vec kernels.
 *
 * <p>Prepared kernels with per-group minima need the sum of each matching input
 * vector group. This helper owns the scratch-buffer reuse and vectorized
 * accumulation without making the main tensor facade carry that detail.</p>
 * @author bhangun
 */
final class GgufSum {
    private GgufSum() {
    }

    static float[] q4KVectorGroupSums(float[] vector, int columns, Q4KWorkBuffer workBuffer) {
        return vectorGroupSums32(vector, columns, workBuffer);
    }

    static float[] vector32GroupSums(float[] vector, int columns, Q4KWorkBuffer workBuffer) {
        return vectorGroupSums32(vector, columns, workBuffer);
    }

    private static float[] vectorGroupSums32(float[] vector, int columns, Q4KWorkBuffer workBuffer) {
        int groups = columns / Q4_0_BLOCK_SIZE;
        float[] sums = workBuffer.vectorGroupSums(groups);
        int vectorBase = 0;
        int group = 0;
        for (; group + 3 < groups; group += 4) {
            sumFloat32Quad(vector, vectorBase, sums, group);
            vectorBase += Q4_0_BLOCK_SIZE * 4;
        }
        for (; group + 1 < groups; group += 2) {
            sumFloat32Pair(vector, vectorBase, sums, group);
            vectorBase += Q4_0_BLOCK_SIZE * 2;
        }
        if (group < groups) {
            sums[group] = sumFloat32(vector, vectorBase);
        }
        return sums;
    }

    static float[] vector16GroupSums(float[] vector, int columns, Q4KWorkBuffer workBuffer) {
        int groups = columns / 16;
        float[] sums = workBuffer.vectorGroupSums(groups);
        int vectorBase = 0;
        int group = 0;
        for (; group + 3 < groups; group += 4) {
            sumFloat16Quad(vector, vectorBase, sums, group);
            vectorBase += 64;
        }
        for (; group + 1 < groups; group += 2) {
            sumFloat16Pair(vector, vectorBase, sums, group);
            vectorBase += 32;
        }
        if (group < groups) {
            sums[group] = sumFloat16(vector, vectorBase);
        }
        return sums;
    }

    private static void sumFloat16Quad(float[] values, int offset, float[] sums, int sumOffset) {
        FloatVector first = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector second = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector third = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector fourth = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        int vectorLimit = 16 - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            first = first.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + i));
            second = second.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + 16 + i));
            third = third.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + 32 + i));
            fourth = fourth.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + 48 + i));
        }
        float firstSum = first.reduceLanes(VectorOperators.ADD);
        float secondSum = second.reduceLanes(VectorOperators.ADD);
        float thirdSum = third.reduceLanes(VectorOperators.ADD);
        float fourthSum = fourth.reduceLanes(VectorOperators.ADD);
        for (; i < 16; i++) {
            firstSum += values[offset + i];
            secondSum += values[offset + 16 + i];
            thirdSum += values[offset + 32 + i];
            fourthSum += values[offset + 48 + i];
        }
        sums[sumOffset] = firstSum;
        sums[sumOffset + 1] = secondSum;
        sums[sumOffset + 2] = thirdSum;
        sums[sumOffset + 3] = fourthSum;
    }

    private static void sumFloat16Pair(float[] values, int offset, float[] sums, int sumOffset) {
        FloatVector first = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector second = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        int vectorLimit = 16 - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            first = first.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + i));
            second = second.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + 16 + i));
        }
        float firstSum = first.reduceLanes(VectorOperators.ADD);
        float secondSum = second.reduceLanes(VectorOperators.ADD);
        for (; i < 16; i++) {
            firstSum += values[offset + i];
            secondSum += values[offset + 16 + i];
        }
        sums[sumOffset] = firstSum;
        sums[sumOffset + 1] = secondSum;
    }

    private static void sumFloat32Quad(float[] values, int offset, float[] sums, int sumOffset) {
        FloatVector first = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector second = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector third = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector fourth = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        int vectorLimit = Q4_0_BLOCK_SIZE - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            first = first.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + i));
            second = second.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + Q4_0_BLOCK_SIZE + i));
            third = third.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + Q4_0_BLOCK_SIZE * 2 + i));
            fourth = fourth.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + Q4_0_BLOCK_SIZE * 3 + i));
        }
        float firstSum = first.reduceLanes(VectorOperators.ADD);
        float secondSum = second.reduceLanes(VectorOperators.ADD);
        float thirdSum = third.reduceLanes(VectorOperators.ADD);
        float fourthSum = fourth.reduceLanes(VectorOperators.ADD);
        for (; i < Q4_0_BLOCK_SIZE; i++) {
            firstSum += values[offset + i];
            secondSum += values[offset + Q4_0_BLOCK_SIZE + i];
            thirdSum += values[offset + Q4_0_BLOCK_SIZE * 2 + i];
            fourthSum += values[offset + Q4_0_BLOCK_SIZE * 3 + i];
        }
        sums[sumOffset] = firstSum;
        sums[sumOffset + 1] = secondSum;
        sums[sumOffset + 2] = thirdSum;
        sums[sumOffset + 3] = fourthSum;
    }

    private static void sumFloat32Pair(float[] values, int offset, float[] sums, int sumOffset) {
        FloatVector first = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector second = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        int vectorLimit = Q4_0_BLOCK_SIZE - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            first = first.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + i));
            second = second.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + Q4_0_BLOCK_SIZE + i));
        }
        float firstSum = first.reduceLanes(VectorOperators.ADD);
        float secondSum = second.reduceLanes(VectorOperators.ADD);
        for (; i < Q4_0_BLOCK_SIZE; i++) {
            firstSum += values[offset + i];
            secondSum += values[offset + Q4_0_BLOCK_SIZE + i];
        }
        sums[sumOffset] = firstSum;
        sums[sumOffset + 1] = secondSum;
    }

    static float sumFloatFixed(float[] values, int offset, int length) {
        return switch (length) {
            case 16 -> sumFloat16(values, offset);
            case Q4_0_BLOCK_SIZE -> sumFloat32(values, offset);
            default -> sumFloat(values, offset, length);
        };
    }

    static float sumFloat16(float[] values, int offset) {
        FloatVector acc = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        int vectorLimit = 16 - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            acc = acc.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + i));
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < 16; i++) {
            sum += values[offset + i];
        }
        return sum;
    }

    static float sumFloat32(float[] values, int offset) {
        FloatVector acc = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        int vectorLimit = Q4_0_BLOCK_SIZE - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            acc = acc.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + i));
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < Q4_0_BLOCK_SIZE; i++) {
            sum += values[offset + i];
        }
        return sum;
    }

    private static float sumFloat(float[] values, int offset, int length) {
        FloatVector acc = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        int vectorLimit = length - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            acc = acc.add(FloatVector.fromArray(FLOAT_SUM_SPECIES, values, offset + i));
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += values[offset + i];
        }
        return sum;
    }
}
