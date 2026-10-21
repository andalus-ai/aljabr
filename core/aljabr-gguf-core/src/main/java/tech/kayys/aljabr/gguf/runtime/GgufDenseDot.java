package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.BF16_DOT_SHORT_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.DENSE_BF16_VECTOR_DOT_ENABLED;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.DENSE_F32_VECTOR_DOT_ENABLED;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.FLOAT_SUM_VECTOR_LANES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.INT_SUM_SPECIES;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Dense raw GGUF row-dot kernels.
 *
 * <p>F32, F16, and BF16 tensors do not need quant deconstruction, but their
 * scalar and Vector API reductions are still hot-path code. Keeping them here
 * separates dense arithmetic from the mixed-format tensor facade.</p>
 * @author bhangun
 */
final class GgufDenseDot {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufDenseDot() {
    }

    static float dotRowF32(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        if (DENSE_F32_VECTOR_DOT_ENABLED && columns >= FLOAT_SUM_VECTOR_LANES) {
            return dotRowF32Vector(segment, rowOffset, columns, vector, vectorOffset);
        }
        return dotRowF32Scalar(segment, rowOffset, columns, vector, vectorOffset);
    }

    static float dotRowF32Vector(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        FloatVector acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        long sourceOffset = rowOffset;
        int vectorBase = vectorOffset;
        int vectorBytes = FLOAT_SUM_VECTOR_LANES * Float.BYTES;
        int unrolledStride = FLOAT_SUM_VECTOR_LANES * 4;
        int unrolledLimit = columns - unrolledStride;
        for (; i <= unrolledLimit; i += unrolledStride) {
            FloatVector weights0 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, sourceOffset, ByteOrder.LITTLE_ENDIAN);
            FloatVector weights1 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, sourceOffset + vectorBytes, ByteOrder.LITTLE_ENDIAN);
            FloatVector weights2 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, sourceOffset + 2L * vectorBytes, ByteOrder.LITTLE_ENDIAN);
            FloatVector weights3 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, sourceOffset + 3L * vectorBytes, ByteOrder.LITTLE_ENDIAN);
            FloatVector input0 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase);
            FloatVector input1 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase + FLOAT_SUM_VECTOR_LANES);
            FloatVector input2 = FloatVector.fromArray(
                    FLOAT_SUM_SPECIES, vector, vectorBase + 2 * FLOAT_SUM_VECTOR_LANES);
            FloatVector input3 = FloatVector.fromArray(
                    FLOAT_SUM_SPECIES, vector, vectorBase + 3 * FLOAT_SUM_VECTOR_LANES);
            acc0 = weights0.fma(input0, acc0);
            acc1 = weights1.fma(input1, acc1);
            acc2 = weights2.fma(input2, acc2);
            acc3 = weights3.fma(input3, acc3);
            sourceOffset += (long) unrolledStride * Float.BYTES;
            vectorBase += unrolledStride;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        int vectorLimit = columns - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            FloatVector weights = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, sourceOffset, ByteOrder.LITTLE_ENDIAN);
            FloatVector input = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase);
            acc = weights.fma(input, acc);
            sourceOffset += vectorBytes;
            vectorBase += FLOAT_SUM_VECTOR_LANES;
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < columns; i++) {
            sum += segment.get(LE_FLOAT, sourceOffset) * vector[vectorBase];
            sourceOffset += Float.BYTES;
            vectorBase++;
        }
        return sum;
    }

    static void dotRowsF32Vector4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + rowBytes + rowBytes;
        long row3Offset = row2Offset + rowBytes;
        long columnOffset = 0L;
        int i = 0;
        int vectorBase = 0;
        int vectorBytes = FLOAT_SUM_VECTOR_LANES * Float.BYTES;
        int vectorLimit = columns - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            FloatVector input = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase);
            acc0 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, row0Offset + columnOffset, ByteOrder.LITTLE_ENDIAN)
                    .fma(input, acc0);
            acc1 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, row1Offset + columnOffset, ByteOrder.LITTLE_ENDIAN)
                    .fma(input, acc1);
            acc2 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, row2Offset + columnOffset, ByteOrder.LITTLE_ENDIAN)
                    .fma(input, acc2);
            acc3 = FloatVector.fromMemorySegment(
                    FLOAT_SUM_SPECIES, segment, row3Offset + columnOffset, ByteOrder.LITTLE_ENDIAN)
                    .fma(input, acc3);
            columnOffset += vectorBytes;
            vectorBase += FLOAT_SUM_VECTOR_LANES;
        }
        float sum0 = acc0.reduceLanes(VectorOperators.ADD);
        float sum1 = acc1.reduceLanes(VectorOperators.ADD);
        float sum2 = acc2.reduceLanes(VectorOperators.ADD);
        float sum3 = acc3.reduceLanes(VectorOperators.ADD);
        for (; i < columns; i++) {
            float input = vector[vectorBase];
            sum0 = Math.fma(segment.get(LE_FLOAT, row0Offset + columnOffset), input, sum0);
            sum1 = Math.fma(segment.get(LE_FLOAT, row1Offset + columnOffset), input, sum1);
            sum2 = Math.fma(segment.get(LE_FLOAT, row2Offset + columnOffset), input, sum2);
            sum3 = Math.fma(segment.get(LE_FLOAT, row3Offset + columnOffset), input, sum3);
            columnOffset += Float.BYTES;
            vectorBase++;
        }
        output[outputOffset] = sum0;
        output[outputOffset + 1] = sum1;
        output[outputOffset + 2] = sum2;
        output[outputOffset + 3] = sum3;
    }

    static float dotRowF32Scalar(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int i = 0;
        long sourceOffset = rowOffset;
        int vectorBase = vectorOffset;
        int unrolledLimit = columns - 16;
        for (; i <= unrolledLimit; i += 16) {
            sum0 += segment.get(LE_FLOAT, sourceOffset) * vector[vectorBase];
            sum1 += segment.get(LE_FLOAT, sourceOffset + 4L) * vector[vectorBase + 1];
            sum2 += segment.get(LE_FLOAT, sourceOffset + 8L) * vector[vectorBase + 2];
            sum3 += segment.get(LE_FLOAT, sourceOffset + 12L) * vector[vectorBase + 3];
            sum0 += segment.get(LE_FLOAT, sourceOffset + 16L) * vector[vectorBase + 4];
            sum1 += segment.get(LE_FLOAT, sourceOffset + 20L) * vector[vectorBase + 5];
            sum2 += segment.get(LE_FLOAT, sourceOffset + 24L) * vector[vectorBase + 6];
            sum3 += segment.get(LE_FLOAT, sourceOffset + 28L) * vector[vectorBase + 7];
            sum0 += segment.get(LE_FLOAT, sourceOffset + 32L) * vector[vectorBase + 8];
            sum1 += segment.get(LE_FLOAT, sourceOffset + 36L) * vector[vectorBase + 9];
            sum2 += segment.get(LE_FLOAT, sourceOffset + 40L) * vector[vectorBase + 10];
            sum3 += segment.get(LE_FLOAT, sourceOffset + 44L) * vector[vectorBase + 11];
            sum0 += segment.get(LE_FLOAT, sourceOffset + 48L) * vector[vectorBase + 12];
            sum1 += segment.get(LE_FLOAT, sourceOffset + 52L) * vector[vectorBase + 13];
            sum2 += segment.get(LE_FLOAT, sourceOffset + 56L) * vector[vectorBase + 14];
            sum3 += segment.get(LE_FLOAT, sourceOffset + 60L) * vector[vectorBase + 15];
            sourceOffset += 16L * Float.BYTES;
            vectorBase += 16;
        }
        int pairLimit = columns - 4;
        for (; i <= pairLimit; i += 4) {
            sum0 += segment.get(LE_FLOAT, sourceOffset) * vector[vectorBase];
            sum1 += segment.get(LE_FLOAT, sourceOffset + 4L) * vector[vectorBase + 1];
            sum2 += segment.get(LE_FLOAT, sourceOffset + 8L) * vector[vectorBase + 2];
            sum3 += segment.get(LE_FLOAT, sourceOffset + 12L) * vector[vectorBase + 3];
            sourceOffset += 4L * Float.BYTES;
            vectorBase += 4;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; i < columns; i++) {
            sum += segment.get(LE_FLOAT, sourceOffset) * vector[vectorBase];
            sourceOffset += Float.BYTES;
            vectorBase++;
        }
        return sum;
    }

    static void dotRowsF32Scalar4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
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
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + rowBytes + rowBytes;
        long row3Offset = row2Offset + rowBytes;
        long columnOffset = 0L;
        int i = 0;
        int vectorBase = 0;
        int pairLimit = columns - 4;
        for (; i <= pairLimit; i += 4) {
            float input0 = vector[vectorBase];
            float input1 = vector[vectorBase + 1];
            float input2 = vector[vectorBase + 2];
            float input3 = vector[vectorBase + 3];
            row0Sum0 += segment.get(LE_FLOAT, row0Offset + columnOffset) * input0;
            row0Sum1 += segment.get(LE_FLOAT, row0Offset + columnOffset + 4L) * input1;
            row0Sum2 += segment.get(LE_FLOAT, row0Offset + columnOffset + 8L) * input2;
            row0Sum3 += segment.get(LE_FLOAT, row0Offset + columnOffset + 12L) * input3;
            row1Sum0 += segment.get(LE_FLOAT, row1Offset + columnOffset) * input0;
            row1Sum1 += segment.get(LE_FLOAT, row1Offset + columnOffset + 4L) * input1;
            row1Sum2 += segment.get(LE_FLOAT, row1Offset + columnOffset + 8L) * input2;
            row1Sum3 += segment.get(LE_FLOAT, row1Offset + columnOffset + 12L) * input3;
            row2Sum0 += segment.get(LE_FLOAT, row2Offset + columnOffset) * input0;
            row2Sum1 += segment.get(LE_FLOAT, row2Offset + columnOffset + 4L) * input1;
            row2Sum2 += segment.get(LE_FLOAT, row2Offset + columnOffset + 8L) * input2;
            row2Sum3 += segment.get(LE_FLOAT, row2Offset + columnOffset + 12L) * input3;
            row3Sum0 += segment.get(LE_FLOAT, row3Offset + columnOffset) * input0;
            row3Sum1 += segment.get(LE_FLOAT, row3Offset + columnOffset + 4L) * input1;
            row3Sum2 += segment.get(LE_FLOAT, row3Offset + columnOffset + 8L) * input2;
            row3Sum3 += segment.get(LE_FLOAT, row3Offset + columnOffset + 12L) * input3;
            columnOffset += 4L * Float.BYTES;
            vectorBase += 4;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; i < columns; i++) {
            float input = vector[vectorBase];
            row0Sum += segment.get(LE_FLOAT, row0Offset + columnOffset) * input;
            row1Sum += segment.get(LE_FLOAT, row1Offset + columnOffset) * input;
            row2Sum += segment.get(LE_FLOAT, row2Offset + columnOffset) * input;
            row3Sum += segment.get(LE_FLOAT, row3Offset + columnOffset) * input;
            columnOffset += Float.BYTES;
            vectorBase++;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowF16(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int i = 0;
        long sourceOffset = rowOffset;
        int vectorBase = vectorOffset;
        int unrolledLimit = columns - 16;
        for (; i <= unrolledLimit; i += 16) {
            sum0 += f16ToF32(segment.get(LE_SHORT, sourceOffset)) * vector[vectorBase];
            sum1 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 2L)) * vector[vectorBase + 1];
            sum2 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 4L)) * vector[vectorBase + 2];
            sum3 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 6L)) * vector[vectorBase + 3];
            sum0 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 8L)) * vector[vectorBase + 4];
            sum1 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 10L)) * vector[vectorBase + 5];
            sum2 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 12L)) * vector[vectorBase + 6];
            sum3 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 14L)) * vector[vectorBase + 7];
            sum0 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 16L)) * vector[vectorBase + 8];
            sum1 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 18L)) * vector[vectorBase + 9];
            sum2 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 20L)) * vector[vectorBase + 10];
            sum3 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 22L)) * vector[vectorBase + 11];
            sum0 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 24L)) * vector[vectorBase + 12];
            sum1 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 26L)) * vector[vectorBase + 13];
            sum2 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 28L)) * vector[vectorBase + 14];
            sum3 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 30L)) * vector[vectorBase + 15];
            sourceOffset += 16L * Short.BYTES;
            vectorBase += 16;
        }
        int pairLimit = columns - 4;
        for (; i <= pairLimit; i += 4) {
            sum0 += f16ToF32(segment.get(LE_SHORT, sourceOffset)) * vector[vectorBase];
            sum1 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 2L)) * vector[vectorBase + 1];
            sum2 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 4L)) * vector[vectorBase + 2];
            sum3 += f16ToF32(segment.get(LE_SHORT, sourceOffset + 6L)) * vector[vectorBase + 3];
            sourceOffset += 4L * Short.BYTES;
            vectorBase += 4;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; i < columns; i++) {
            sum += f16ToF32(segment.get(LE_SHORT, sourceOffset)) * vector[vectorBase];
            sourceOffset += Short.BYTES;
            vectorBase++;
        }
        return sum;
    }

    static void dotRowsF16Scalar4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
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
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + rowBytes + rowBytes;
        long row3Offset = row2Offset + rowBytes;
        long columnOffset = 0L;
        int i = 0;
        int vectorBase = 0;
        int unrolledLimit = columns - 16;
        for (; i <= unrolledLimit; i += 16) {
            float input0 = vector[vectorBase];
            float input1 = vector[vectorBase + 1];
            float input2 = vector[vectorBase + 2];
            float input3 = vector[vectorBase + 3];
            row0Sum0 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset)) * input0;
            row0Sum1 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 2L)) * input1;
            row0Sum2 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 4L)) * input2;
            row0Sum3 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 6L)) * input3;
            row1Sum0 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset)) * input0;
            row1Sum1 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 2L)) * input1;
            row1Sum2 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 4L)) * input2;
            row1Sum3 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 6L)) * input3;
            row2Sum0 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset)) * input0;
            row2Sum1 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 2L)) * input1;
            row2Sum2 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 4L)) * input2;
            row2Sum3 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 6L)) * input3;
            row3Sum0 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset)) * input0;
            row3Sum1 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 2L)) * input1;
            row3Sum2 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 4L)) * input2;
            row3Sum3 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 6L)) * input3;

            float input4 = vector[vectorBase + 4];
            float input5 = vector[vectorBase + 5];
            float input6 = vector[vectorBase + 6];
            float input7 = vector[vectorBase + 7];
            row0Sum0 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 8L)) * input4;
            row0Sum1 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 10L)) * input5;
            row0Sum2 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 12L)) * input6;
            row0Sum3 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 14L)) * input7;
            row1Sum0 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 8L)) * input4;
            row1Sum1 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 10L)) * input5;
            row1Sum2 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 12L)) * input6;
            row1Sum3 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 14L)) * input7;
            row2Sum0 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 8L)) * input4;
            row2Sum1 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 10L)) * input5;
            row2Sum2 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 12L)) * input6;
            row2Sum3 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 14L)) * input7;
            row3Sum0 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 8L)) * input4;
            row3Sum1 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 10L)) * input5;
            row3Sum2 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 12L)) * input6;
            row3Sum3 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 14L)) * input7;

            float input8 = vector[vectorBase + 8];
            float input9 = vector[vectorBase + 9];
            float input10 = vector[vectorBase + 10];
            float input11 = vector[vectorBase + 11];
            row0Sum0 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 16L)) * input8;
            row0Sum1 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 18L)) * input9;
            row0Sum2 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 20L)) * input10;
            row0Sum3 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 22L)) * input11;
            row1Sum0 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 16L)) * input8;
            row1Sum1 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 18L)) * input9;
            row1Sum2 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 20L)) * input10;
            row1Sum3 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 22L)) * input11;
            row2Sum0 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 16L)) * input8;
            row2Sum1 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 18L)) * input9;
            row2Sum2 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 20L)) * input10;
            row2Sum3 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 22L)) * input11;
            row3Sum0 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 16L)) * input8;
            row3Sum1 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 18L)) * input9;
            row3Sum2 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 20L)) * input10;
            row3Sum3 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 22L)) * input11;

            float input12 = vector[vectorBase + 12];
            float input13 = vector[vectorBase + 13];
            float input14 = vector[vectorBase + 14];
            float input15 = vector[vectorBase + 15];
            row0Sum0 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 24L)) * input12;
            row0Sum1 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 26L)) * input13;
            row0Sum2 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 28L)) * input14;
            row0Sum3 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 30L)) * input15;
            row1Sum0 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 24L)) * input12;
            row1Sum1 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 26L)) * input13;
            row1Sum2 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 28L)) * input14;
            row1Sum3 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 30L)) * input15;
            row2Sum0 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 24L)) * input12;
            row2Sum1 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 26L)) * input13;
            row2Sum2 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 28L)) * input14;
            row2Sum3 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 30L)) * input15;
            row3Sum0 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 24L)) * input12;
            row3Sum1 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 26L)) * input13;
            row3Sum2 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 28L)) * input14;
            row3Sum3 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 30L)) * input15;

            columnOffset += 16L * Short.BYTES;
            vectorBase += 16;
        }
        int pairLimit = columns - 4;
        for (; i <= pairLimit; i += 4) {
            float input0 = vector[vectorBase];
            float input1 = vector[vectorBase + 1];
            float input2 = vector[vectorBase + 2];
            float input3 = vector[vectorBase + 3];
            row0Sum0 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset)) * input0;
            row0Sum1 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 2L)) * input1;
            row0Sum2 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 4L)) * input2;
            row0Sum3 += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 6L)) * input3;
            row1Sum0 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset)) * input0;
            row1Sum1 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 2L)) * input1;
            row1Sum2 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 4L)) * input2;
            row1Sum3 += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 6L)) * input3;
            row2Sum0 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset)) * input0;
            row2Sum1 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 2L)) * input1;
            row2Sum2 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 4L)) * input2;
            row2Sum3 += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 6L)) * input3;
            row3Sum0 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset)) * input0;
            row3Sum1 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 2L)) * input1;
            row3Sum2 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 4L)) * input2;
            row3Sum3 += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 6L)) * input3;
            columnOffset += 4L * Short.BYTES;
            vectorBase += 4;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; i < columns; i++) {
            float input = vector[vectorBase];
            row0Sum += f16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset)) * input;
            row1Sum += f16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset)) * input;
            row2Sum += f16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset)) * input;
            row3Sum += f16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset)) * input;
            columnOffset += Short.BYTES;
            vectorBase++;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowBF16(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        if (DENSE_BF16_VECTOR_DOT_ENABLED && columns >= FLOAT_SUM_VECTOR_LANES) {
            return dotRowBF16Vector(segment, rowOffset, columns, vector, vectorOffset);
        }
        return dotRowBF16Scalar(segment, rowOffset, columns, vector, vectorOffset);
    }

    static float dotRowBF16Vector(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        FloatVector acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        int i = 0;
        long sourceOffset = rowOffset;
        int vectorBase = vectorOffset;
        int vectorBytes = FLOAT_SUM_VECTOR_LANES * Short.BYTES;
        int unrolledStride = FLOAT_SUM_VECTOR_LANES * 4;
        int unrolledLimit = columns - unrolledStride;
        for (; i <= unrolledLimit; i += unrolledStride) {
            FloatVector weights0 = bf16Vector(segment, sourceOffset);
            FloatVector weights1 = bf16Vector(segment, sourceOffset + vectorBytes);
            FloatVector weights2 = bf16Vector(segment, sourceOffset + 2L * vectorBytes);
            FloatVector weights3 = bf16Vector(segment, sourceOffset + 3L * vectorBytes);
            FloatVector input0 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase);
            FloatVector input1 = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase + FLOAT_SUM_VECTOR_LANES);
            FloatVector input2 = FloatVector.fromArray(
                    FLOAT_SUM_SPECIES, vector, vectorBase + 2 * FLOAT_SUM_VECTOR_LANES);
            FloatVector input3 = FloatVector.fromArray(
                    FLOAT_SUM_SPECIES, vector, vectorBase + 3 * FLOAT_SUM_VECTOR_LANES);
            acc0 = weights0.fma(input0, acc0);
            acc1 = weights1.fma(input1, acc1);
            acc2 = weights2.fma(input2, acc2);
            acc3 = weights3.fma(input3, acc3);
            sourceOffset += (long) unrolledStride * Short.BYTES;
            vectorBase += unrolledStride;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        int vectorLimit = columns - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            FloatVector weights = bf16Vector(segment, sourceOffset);
            FloatVector input = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase);
            acc = weights.fma(input, acc);
            sourceOffset += vectorBytes;
            vectorBase += FLOAT_SUM_VECTOR_LANES;
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < columns; i++) {
            sum += bf16ToF32(segment.get(LE_SHORT, sourceOffset)) * vector[vectorBase];
            sourceOffset += Short.BYTES;
            vectorBase++;
        }
        return sum;
    }

    static void dotRowsBF16Vector4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        FloatVector acc0 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc1 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc2 = FloatVector.zero(FLOAT_SUM_SPECIES);
        FloatVector acc3 = FloatVector.zero(FLOAT_SUM_SPECIES);
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + rowBytes + rowBytes;
        long row3Offset = row2Offset + rowBytes;
        long columnOffset = 0L;
        int i = 0;
        int vectorBase = 0;
        int vectorBytes = FLOAT_SUM_VECTOR_LANES * Short.BYTES;
        int vectorLimit = columns - FLOAT_SUM_VECTOR_LANES;
        for (; i <= vectorLimit; i += FLOAT_SUM_VECTOR_LANES) {
            FloatVector input = FloatVector.fromArray(FLOAT_SUM_SPECIES, vector, vectorBase);
            acc0 = bf16Vector(segment, row0Offset + columnOffset).fma(input, acc0);
            acc1 = bf16Vector(segment, row1Offset + columnOffset).fma(input, acc1);
            acc2 = bf16Vector(segment, row2Offset + columnOffset).fma(input, acc2);
            acc3 = bf16Vector(segment, row3Offset + columnOffset).fma(input, acc3);
            columnOffset += vectorBytes;
            vectorBase += FLOAT_SUM_VECTOR_LANES;
        }
        float sum0 = acc0.reduceLanes(VectorOperators.ADD);
        float sum1 = acc1.reduceLanes(VectorOperators.ADD);
        float sum2 = acc2.reduceLanes(VectorOperators.ADD);
        float sum3 = acc3.reduceLanes(VectorOperators.ADD);
        for (; i < columns; i++) {
            float input = vector[vectorBase];
            sum0 = Math.fma(bf16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset)), input, sum0);
            sum1 = Math.fma(bf16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset)), input, sum1);
            sum2 = Math.fma(bf16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset)), input, sum2);
            sum3 = Math.fma(bf16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset)), input, sum3);
            columnOffset += Short.BYTES;
            vectorBase++;
        }
        output[outputOffset] = sum0;
        output[outputOffset + 1] = sum1;
        output[outputOffset + 2] = sum2;
        output[outputOffset + 3] = sum3;
    }

    private static FloatVector bf16Vector(MemorySegment segment, long sourceOffset) {
        ShortVector raw = ShortVector.fromMemorySegment(
                BF16_DOT_SHORT_SPECIES, segment, sourceOffset, ByteOrder.LITTLE_ENDIAN);
        IntVector bits = (IntVector) raw.convertShape(VectorOperators.ZERO_EXTEND_S2I, INT_SUM_SPECIES, 0);
        return bits.lanewise(VectorOperators.LSHL, 16).reinterpretAsFloats();
    }

    private static float bf16ToF32(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    static float dotRowBF16Scalar(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int i = 0;
        long sourceOffset = rowOffset;
        int vectorBase = vectorOffset;
        int unrolledLimit = columns - 16;
        for (; i <= unrolledLimit; i += 16) {
            sum0 += bf16ToF32(segment.get(LE_SHORT, sourceOffset)) * vector[vectorBase];
            sum1 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 2L)) * vector[vectorBase + 1];
            sum2 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 4L)) * vector[vectorBase + 2];
            sum3 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 6L)) * vector[vectorBase + 3];
            sum0 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 8L)) * vector[vectorBase + 4];
            sum1 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 10L)) * vector[vectorBase + 5];
            sum2 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 12L)) * vector[vectorBase + 6];
            sum3 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 14L)) * vector[vectorBase + 7];
            sum0 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 16L)) * vector[vectorBase + 8];
            sum1 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 18L)) * vector[vectorBase + 9];
            sum2 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 20L)) * vector[vectorBase + 10];
            sum3 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 22L)) * vector[vectorBase + 11];
            sum0 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 24L)) * vector[vectorBase + 12];
            sum1 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 26L)) * vector[vectorBase + 13];
            sum2 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 28L)) * vector[vectorBase + 14];
            sum3 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 30L)) * vector[vectorBase + 15];
            sourceOffset += 16L * Short.BYTES;
            vectorBase += 16;
        }
        int pairLimit = columns - 4;
        for (; i <= pairLimit; i += 4) {
            sum0 += bf16ToF32(segment.get(LE_SHORT, sourceOffset)) * vector[vectorBase];
            sum1 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 2L)) * vector[vectorBase + 1];
            sum2 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 4L)) * vector[vectorBase + 2];
            sum3 += bf16ToF32(segment.get(LE_SHORT, sourceOffset + 6L)) * vector[vectorBase + 3];
            sourceOffset += 4L * Short.BYTES;
            vectorBase += 4;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; i < columns; i++) {
            sum += bf16ToF32(segment.get(LE_SHORT, sourceOffset)) * vector[vectorBase];
            sourceOffset += Short.BYTES;
            vectorBase++;
        }
        return sum;
    }

    static void dotRowsBF16Scalar4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
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
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + rowBytes + rowBytes;
        long row3Offset = row2Offset + rowBytes;
        long columnOffset = 0L;
        int i = 0;
        int vectorBase = 0;
        int pairLimit = columns - 4;
        for (; i <= pairLimit; i += 4) {
            float input0 = vector[vectorBase];
            float input1 = vector[vectorBase + 1];
            float input2 = vector[vectorBase + 2];
            float input3 = vector[vectorBase + 3];
            row0Sum0 += bf16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset)) * input0;
            row0Sum1 += bf16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 2L)) * input1;
            row0Sum2 += bf16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 4L)) * input2;
            row0Sum3 += bf16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset + 6L)) * input3;
            row1Sum0 += bf16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset)) * input0;
            row1Sum1 += bf16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 2L)) * input1;
            row1Sum2 += bf16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 4L)) * input2;
            row1Sum3 += bf16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset + 6L)) * input3;
            row2Sum0 += bf16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset)) * input0;
            row2Sum1 += bf16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 2L)) * input1;
            row2Sum2 += bf16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 4L)) * input2;
            row2Sum3 += bf16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset + 6L)) * input3;
            row3Sum0 += bf16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset)) * input0;
            row3Sum1 += bf16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 2L)) * input1;
            row3Sum2 += bf16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 4L)) * input2;
            row3Sum3 += bf16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset + 6L)) * input3;
            columnOffset += 4L * Short.BYTES;
            vectorBase += 4;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; i < columns; i++) {
            float input = vector[vectorBase];
            row0Sum += bf16ToF32(segment.get(LE_SHORT, row0Offset + columnOffset)) * input;
            row1Sum += bf16ToF32(segment.get(LE_SHORT, row1Offset + columnOffset)) * input;
            row2Sum += bf16ToF32(segment.get(LE_SHORT, row2Offset + columnOffset)) * input;
            row3Sum += bf16ToF32(segment.get(LE_SHORT, row3Offset + columnOffset)) * input;
            columnOffset += Short.BYTES;
            vectorBase++;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }
}
