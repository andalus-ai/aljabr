package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.signedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q8_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.Q4_DOT_BYTE_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.Q4_DOT_FLOAT_SPECIES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.Q4_DOT_VECTOR_LANES;
import static tech.kayys.aljabr.gguf.runtime.GgufVectorConfig.RAW_Q8_VECTOR_DOT_ENABLED;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Raw Q8 GGUF row-dot kernels.
 *
 * <p>Q8_0, Q8_1, and Q8_K all reduce signed byte quant blocks with per-block
 * scales. This helper keeps their scalar and Vector API kernels together.</p>
 * @author bhangun
 */
final class GgufQ8RawDot {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ8RawDot() {
    }

    static float dotRowQ8_0(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        if (RAW_Q8_VECTOR_DOT_ENABLED) {
            return dotRowQ8_0Vector(segment, rowOffset, columns, vector, vectorOffset);
        }
        return dotRowQ8_0Scalar(segment, rowOffset, columns, vector, vectorOffset);
    }

    static float dotRowQ8_1(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        if (RAW_Q8_VECTOR_DOT_ENABLED) {
            return dotRowQ8_1Vector(segment, rowOffset, columns, vector, vectorOffset);
        }
        return dotRowQ8_1Scalar(segment, rowOffset, columns, vector, vectorOffset);
    }

    static float dotRowQ8_0Vector(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / Q8_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            acc0 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + Short.BYTES,
                    vector,
                    vectorBase,
                    f16ToF32(segment.get(LE_SHORT, blockOffset)),
                    acc0);
            acc1 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + Q8_0_BLOCK_BYTES + Short.BYTES,
                    vector,
                    vectorBase + Q8_0_BLOCK_SIZE,
                    f16ToF32(segment.get(LE_SHORT, blockOffset + Q8_0_BLOCK_BYTES)),
                    acc1);
            acc2 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + 2L * Q8_0_BLOCK_BYTES + Short.BYTES,
                    vector,
                    vectorBase + 2 * Q8_0_BLOCK_SIZE,
                    f16ToF32(segment.get(LE_SHORT, blockOffset + 2L * Q8_0_BLOCK_BYTES)),
                    acc2);
            acc3 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + 3L * Q8_0_BLOCK_BYTES + Short.BYTES,
                    vector,
                    vectorBase + 3 * Q8_0_BLOCK_SIZE,
                    f16ToF32(segment.get(LE_SHORT, blockOffset + 3L * Q8_0_BLOCK_BYTES)),
                    acc3);
            blockOffset += 4L * Q8_0_BLOCK_BYTES;
            vectorBase += 4 * Q8_0_BLOCK_SIZE;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        for (; block < blocks; block++) {
            acc = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + Short.BYTES,
                    vector,
                    vectorBase,
                    f16ToF32(segment.get(LE_SHORT, blockOffset)),
                    acc);
            blockOffset += Q8_0_BLOCK_BYTES;
            vectorBase += Q8_0_BLOCK_SIZE;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    static void dotRowsQ8_0Vector4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotRowsQ8Vector4(
                segment, rowOffset, rowBytes, columns, vector, output, outputOffset, Q8_0_BLOCK_BYTES, Short.BYTES);
    }

    static float dotRowQ8_0Scalar(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / Q8_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += f16ToF32(segment.get(LE_SHORT, blockOffset))
                    * dotRowQ8UnscaledBlockScalar(segment, blockOffset + Short.BYTES, vector, vectorBase);
            sum1 += f16ToF32(segment.get(LE_SHORT, blockOffset + Q8_0_BLOCK_BYTES))
                    * dotRowQ8UnscaledBlockScalar(
                            segment,
                            blockOffset + Q8_0_BLOCK_BYTES + Short.BYTES,
                            vector,
                            vectorBase + Q8_0_BLOCK_SIZE);
            sum2 += f16ToF32(segment.get(LE_SHORT, blockOffset + 2L * Q8_0_BLOCK_BYTES))
                    * dotRowQ8UnscaledBlockScalar(
                            segment,
                            blockOffset + 2L * Q8_0_BLOCK_BYTES + Short.BYTES,
                            vector,
                            vectorBase + 2 * Q8_0_BLOCK_SIZE);
            sum3 += f16ToF32(segment.get(LE_SHORT, blockOffset + 3L * Q8_0_BLOCK_BYTES))
                    * dotRowQ8UnscaledBlockScalar(
                            segment,
                            blockOffset + 3L * Q8_0_BLOCK_BYTES + Short.BYTES,
                            vector,
                            vectorBase + 3 * Q8_0_BLOCK_SIZE);
            blockOffset += 4L * Q8_0_BLOCK_BYTES;
            vectorBase += 4 * Q8_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            sum += d * dotRowQ8UnscaledBlockScalar(segment, blockOffset + Short.BYTES, vector, vectorBase);
            blockOffset += Q8_0_BLOCK_BYTES;
            vectorBase += Q8_0_BLOCK_SIZE;
        }
        return sum;
    }

    static void dotRowsQ8_0Scalar4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotRowsQ8Scalar4(
                segment, rowOffset, rowBytes, columns, vector, output, outputOffset, Q8_0_BLOCK_BYTES, Short.BYTES);
    }

    static float dotRowQ8_1Vector(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / Q8_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            acc0 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + 2L * Short.BYTES,
                    vector,
                    vectorBase,
                    f16ToF32(segment.get(LE_SHORT, blockOffset)),
                    acc0);
            acc1 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + Q8_1_BLOCK_BYTES + 2L * Short.BYTES,
                    vector,
                    vectorBase + Q8_0_BLOCK_SIZE,
                    f16ToF32(segment.get(LE_SHORT, blockOffset + Q8_1_BLOCK_BYTES)),
                    acc1);
            acc2 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + 2L * Q8_1_BLOCK_BYTES + 2L * Short.BYTES,
                    vector,
                    vectorBase + 2 * Q8_0_BLOCK_SIZE,
                    f16ToF32(segment.get(LE_SHORT, blockOffset + 2L * Q8_1_BLOCK_BYTES)),
                    acc2);
            acc3 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + 3L * Q8_1_BLOCK_BYTES + 2L * Short.BYTES,
                    vector,
                    vectorBase + 3 * Q8_0_BLOCK_SIZE,
                    f16ToF32(segment.get(LE_SHORT, blockOffset + 3L * Q8_1_BLOCK_BYTES)),
                    acc3);
            blockOffset += 4L * Q8_1_BLOCK_BYTES;
            vectorBase += 4 * Q8_0_BLOCK_SIZE;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        for (; block < blocks; block++) {
            acc = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    blockOffset + 2L * Short.BYTES,
                    vector,
                    vectorBase,
                    f16ToF32(segment.get(LE_SHORT, blockOffset)),
                    acc);
            blockOffset += Q8_1_BLOCK_BYTES;
            vectorBase += Q8_0_BLOCK_SIZE;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    static void dotRowsQ8_1Vector4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotRowsQ8Vector4(
                segment,
                rowOffset,
                rowBytes,
                columns,
                vector,
                output,
                outputOffset,
                Q8_1_BLOCK_BYTES,
                2L * Short.BYTES);
    }

    static float dotRowQ8_1Scalar(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / Q8_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += f16ToF32(segment.get(LE_SHORT, blockOffset))
                    * dotRowQ8UnscaledBlockScalar(
                            segment,
                            blockOffset + 2L * Short.BYTES,
                            vector,
                            vectorBase);
            sum1 += f16ToF32(segment.get(LE_SHORT, blockOffset + Q8_1_BLOCK_BYTES))
                    * dotRowQ8UnscaledBlockScalar(
                            segment,
                            blockOffset + Q8_1_BLOCK_BYTES + 2L * Short.BYTES,
                            vector,
                            vectorBase + Q8_0_BLOCK_SIZE);
            sum2 += f16ToF32(segment.get(LE_SHORT, blockOffset + 2L * Q8_1_BLOCK_BYTES))
                    * dotRowQ8UnscaledBlockScalar(
                            segment,
                            blockOffset + 2L * Q8_1_BLOCK_BYTES + 2L * Short.BYTES,
                            vector,
                            vectorBase + 2 * Q8_0_BLOCK_SIZE);
            sum3 += f16ToF32(segment.get(LE_SHORT, blockOffset + 3L * Q8_1_BLOCK_BYTES))
                    * dotRowQ8UnscaledBlockScalar(
                            segment,
                            blockOffset + 3L * Q8_1_BLOCK_BYTES + 2L * Short.BYTES,
                            vector,
                            vectorBase + 3 * Q8_0_BLOCK_SIZE);
            blockOffset += 4L * Q8_1_BLOCK_BYTES;
            vectorBase += 4 * Q8_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            sum += d * dotRowQ8UnscaledBlockScalar(
                    segment, blockOffset + 2L * Short.BYTES, vector, vectorBase);
            blockOffset += Q8_1_BLOCK_BYTES;
            vectorBase += Q8_0_BLOCK_SIZE;
        }
        return sum;
    }

    static void dotRowsQ8_1Scalar4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        dotRowsQ8Scalar4(
                segment,
                rowOffset,
                rowBytes,
                columns,
                vector,
                output,
                outputOffset,
                Q8_1_BLOCK_BYTES,
                2L * Short.BYTES);
    }

    private static void dotRowsQ8Scalar4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset,
            int blockBytes,
            long quantOffset) {
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
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / Q8_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotScaledQ8ScalarBlock(segment, row0Offset + blockOffset, quantOffset, vector, vectorBase);
            row1Sum0 += dotScaledQ8ScalarBlock(segment, row1Offset + blockOffset, quantOffset, vector, vectorBase);
            row2Sum0 += dotScaledQ8ScalarBlock(segment, row2Offset + blockOffset, quantOffset, vector, vectorBase);
            row3Sum0 += dotScaledQ8ScalarBlock(segment, row3Offset + blockOffset, quantOffset, vector, vectorBase);
            row0Sum1 += dotScaledQ8ScalarBlock(
                    segment,
                    row0Offset + blockOffset + blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + Q8_0_BLOCK_SIZE);
            row1Sum1 += dotScaledQ8ScalarBlock(
                    segment,
                    row1Offset + blockOffset + blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + Q8_0_BLOCK_SIZE);
            row2Sum1 += dotScaledQ8ScalarBlock(
                    segment,
                    row2Offset + blockOffset + blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + Q8_0_BLOCK_SIZE);
            row3Sum1 += dotScaledQ8ScalarBlock(
                    segment,
                    row3Offset + blockOffset + blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + Q8_0_BLOCK_SIZE);
            row0Sum2 += dotScaledQ8ScalarBlock(
                    segment,
                    row0Offset + blockOffset + 2L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 2 * Q8_0_BLOCK_SIZE);
            row1Sum2 += dotScaledQ8ScalarBlock(
                    segment,
                    row1Offset + blockOffset + 2L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 2 * Q8_0_BLOCK_SIZE);
            row2Sum2 += dotScaledQ8ScalarBlock(
                    segment,
                    row2Offset + blockOffset + 2L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 2 * Q8_0_BLOCK_SIZE);
            row3Sum2 += dotScaledQ8ScalarBlock(
                    segment,
                    row3Offset + blockOffset + 2L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 2 * Q8_0_BLOCK_SIZE);
            row0Sum3 += dotScaledQ8ScalarBlock(
                    segment,
                    row0Offset + blockOffset + 3L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 3 * Q8_0_BLOCK_SIZE);
            row1Sum3 += dotScaledQ8ScalarBlock(
                    segment,
                    row1Offset + blockOffset + 3L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 3 * Q8_0_BLOCK_SIZE);
            row2Sum3 += dotScaledQ8ScalarBlock(
                    segment,
                    row2Offset + blockOffset + 3L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 3 * Q8_0_BLOCK_SIZE);
            row3Sum3 += dotScaledQ8ScalarBlock(
                    segment,
                    row3Offset + blockOffset + 3L * blockBytes,
                    quantOffset,
                    vector,
                    vectorBase + 3 * Q8_0_BLOCK_SIZE);
            blockOffset += 4L * blockBytes;
            vectorBase += 4 * Q8_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotScaledQ8ScalarBlock(segment, row0Offset + blockOffset, quantOffset, vector, vectorBase);
            row1Sum += dotScaledQ8ScalarBlock(segment, row1Offset + blockOffset, quantOffset, vector, vectorBase);
            row2Sum += dotScaledQ8ScalarBlock(segment, row2Offset + blockOffset, quantOffset, vector, vectorBase);
            row3Sum += dotScaledQ8ScalarBlock(segment, row3Offset + blockOffset, quantOffset, vector, vectorBase);
            blockOffset += blockBytes;
            vectorBase += Q8_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotScaledQ8ScalarBlock(
            MemorySegment segment,
            long blockOffset,
            long quantOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        return d * dotRowQ8UnscaledBlockScalar(segment, blockOffset + quantOffset, vector, vectorBase);
    }

    static float dotRowQ8UnscaledBlockVector(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset) {
        FloatVector acc = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        return accumulateQ8UnscaledBlockVector(segment, quantsOffset, vector, vectorOffset, acc)
                .reduceLanes(VectorOperators.ADD);
    }

    private static void dotRowsQ8Vector4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset,
            int blockBytes,
            long quantOffset) {
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
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + rowBytes + rowBytes;
        long row3Offset = row2Offset + rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / Q8_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vectorBase);
            FloatVector vf1 = FloatVector.fromArray(
                    Q4_DOT_FLOAT_SPECIES, vector, vectorBase + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(
                    Q4_DOT_FLOAT_SPECIES, vector, vectorBase + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(
                    Q4_DOT_FLOAT_SPECIES, vector, vectorBase + 3 * Q4_DOT_VECTOR_LANES);
            long row0Block = row0Offset + blockOffset;
            long row1Block = row1Offset + blockOffset;
            long row2Block = row2Offset + blockOffset;
            long row3Block = row3Offset + blockOffset;
            row0Acc0 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    row0Block + quantOffset,
                    vf0,
                    vf1,
                    vf2,
                    vf3,
                    f16ToF32(segment.get(LE_SHORT, row0Block)),
                    row0Acc0);
            row1Acc0 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    row1Block + quantOffset,
                    vf0,
                    vf1,
                    vf2,
                    vf3,
                    f16ToF32(segment.get(LE_SHORT, row1Block)),
                    row1Acc0);
            row2Acc0 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    row2Block + quantOffset,
                    vf0,
                    vf1,
                    vf2,
                    vf3,
                    f16ToF32(segment.get(LE_SHORT, row2Block)),
                    row2Acc0);
            row3Acc0 = accumulateScaledQ8UnscaledBlockVector(
                    segment,
                    row3Block + quantOffset,
                    vf0,
                    vf1,
                    vf2,
                    vf3,
                    f16ToF32(segment.get(LE_SHORT, row3Block)),
                    row3Acc0);

            int vector1 = vectorBase + Q8_0_BLOCK_SIZE;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector1 + 3 * Q4_DOT_VECTOR_LANES);
            row0Block += blockBytes;
            row1Block += blockBytes;
            row2Block += blockBytes;
            row3Block += blockBytes;
            row0Acc1 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row0Block)), row0Acc1);
            row1Acc1 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row1Block)), row1Acc1);
            row2Acc1 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row2Block)), row2Acc1);
            row3Acc1 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row3Block)), row3Acc1);

            int vector2 = vectorBase + 2 * Q8_0_BLOCK_SIZE;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector2 + 3 * Q4_DOT_VECTOR_LANES);
            row0Block += blockBytes;
            row1Block += blockBytes;
            row2Block += blockBytes;
            row3Block += blockBytes;
            row0Acc2 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row0Block)), row0Acc2);
            row1Acc2 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row1Block)), row1Acc2);
            row2Acc2 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row2Block)), row2Acc2);
            row3Acc2 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row3Block)), row3Acc2);

            int vector3 = vectorBase + 3 * Q8_0_BLOCK_SIZE;
            vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3);
            vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + Q4_DOT_VECTOR_LANES);
            vf2 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + 2 * Q4_DOT_VECTOR_LANES);
            vf3 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vector3 + 3 * Q4_DOT_VECTOR_LANES);
            row0Block += blockBytes;
            row1Block += blockBytes;
            row2Block += blockBytes;
            row3Block += blockBytes;
            row0Acc3 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row0Block)), row0Acc3);
            row1Acc3 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row1Block)), row1Acc3);
            row2Acc3 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row2Block)), row2Acc3);
            row3Acc3 = accumulateScaledQ8UnscaledBlockVector(
                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row3Block)), row3Acc3);

            blockOffset += 4L * blockBytes;
            vectorBase += 4 * Q8_0_BLOCK_SIZE;
        }
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        for (; block < blocks; block++) {
            FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vectorBase);
            FloatVector vf1 = FloatVector.fromArray(
                    Q4_DOT_FLOAT_SPECIES, vector, vectorBase + Q4_DOT_VECTOR_LANES);
            FloatVector vf2 = FloatVector.fromArray(
                    Q4_DOT_FLOAT_SPECIES, vector, vectorBase + 2 * Q4_DOT_VECTOR_LANES);
            FloatVector vf3 = FloatVector.fromArray(
                    Q4_DOT_FLOAT_SPECIES, vector, vectorBase + 3 * Q4_DOT_VECTOR_LANES);
            long row0Block = row0Offset + blockOffset;
            long row1Block = row1Offset + blockOffset;
            long row2Block = row2Offset + blockOffset;
            long row3Block = row3Offset + blockOffset;
            row0Acc = accumulateScaledQ8UnscaledBlockVector(
                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row0Block)), row0Acc);
            row1Acc = accumulateScaledQ8UnscaledBlockVector(
                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row1Block)), row1Acc);
            row2Acc = accumulateScaledQ8UnscaledBlockVector(
                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row2Block)), row2Acc);
            row3Acc = accumulateScaledQ8UnscaledBlockVector(
                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3,
                    f16ToF32(segment.get(LE_SHORT, row3Block)), row3Acc);
            blockOffset += blockBytes;
            vectorBase += Q8_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 1] = row1Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 2] = row2Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 3] = row3Acc.reduceLanes(VectorOperators.ADD);
    }

    private static FloatVector accumulateQ8UnscaledBlockVector(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset,
            FloatVector acc) {
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vectorOffset);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vectorOffset + Q4_DOT_VECTOR_LANES);
        FloatVector vf2 = FloatVector.fromArray(
                Q4_DOT_FLOAT_SPECIES, vector, vectorOffset + 2 * Q4_DOT_VECTOR_LANES);
        FloatVector vf3 = FloatVector.fromArray(
                Q4_DOT_FLOAT_SPECIES, vector, vectorOffset + 3 * Q4_DOT_VECTOR_LANES);
        return accumulateQ8UnscaledBlockVector(segment, quantsOffset, vf0, vf1, vf2, vf3, acc);
    }

    private static FloatVector accumulateQ8UnscaledBlockVector(
            MemorySegment segment,
            long quantsOffset,
            FloatVector vf0,
            FloatVector vf1,
            FloatVector vf2,
            FloatVector vf3,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset, ByteOrder.LITTLE_ENDIAN);
        ByteVector q1 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset + Q4_DOT_VECTOR_LANES, ByteOrder.LITTLE_ENDIAN);
        ByteVector q2 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset + 2L * Q4_DOT_VECTOR_LANES, ByteOrder.LITTLE_ENDIAN);
        ByteVector q3 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset + 3L * Q4_DOT_VECTOR_LANES, ByteOrder.LITTLE_ENDIAN);
        FloatVector qf0 = (FloatVector) q0.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf1 = (FloatVector) q1.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf2 = (FloatVector) q2.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        FloatVector qf3 = (FloatVector) q3.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
        acc = qf0.fma(vf0, acc);
        acc = qf1.fma(vf1, acc);
        acc = qf2.fma(vf2, acc);
        return qf3.fma(vf3, acc);
    }

    private static FloatVector accumulateScaledQ8UnscaledBlockVector(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset,
            float scale,
            FloatVector acc) {
        FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vectorOffset);
        FloatVector vf1 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vectorOffset + Q4_DOT_VECTOR_LANES);
        FloatVector vf2 = FloatVector.fromArray(
                Q4_DOT_FLOAT_SPECIES, vector, vectorOffset + 2 * Q4_DOT_VECTOR_LANES);
        FloatVector vf3 = FloatVector.fromArray(
                Q4_DOT_FLOAT_SPECIES, vector, vectorOffset + 3 * Q4_DOT_VECTOR_LANES);
        return accumulateScaledQ8UnscaledBlockVector(segment, quantsOffset, vf0, vf1, vf2, vf3, scale, acc);
    }

    private static FloatVector accumulateScaledQ8UnscaledBlockVector(
            MemorySegment segment,
            long quantsOffset,
            FloatVector vf0,
            FloatVector vf1,
            FloatVector vf2,
            FloatVector vf3,
            float scale,
            FloatVector acc) {
        ByteVector q0 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset, ByteOrder.LITTLE_ENDIAN);
        ByteVector q1 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset + Q4_DOT_VECTOR_LANES, ByteOrder.LITTLE_ENDIAN);
        ByteVector q2 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset + 2L * Q4_DOT_VECTOR_LANES, ByteOrder.LITTLE_ENDIAN);
        ByteVector q3 = ByteVector.fromMemorySegment(
                Q4_DOT_BYTE_SPECIES, segment, quantsOffset + 3L * Q4_DOT_VECTOR_LANES, ByteOrder.LITTLE_ENDIAN);
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

    static float dotRowQ8UnscaledBlockScalar(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset) {
        long packed0 = segment.get(LE_LONG, quantsOffset);
        long packed1 = segment.get(LE_LONG, quantsOffset + Long.BYTES);
        long packed2 = segment.get(LE_LONG, quantsOffset + 2L * Long.BYTES);
        long packed3 = segment.get(LE_LONG, quantsOffset + 3L * Long.BYTES);
        float sum0 = signedByte(packed0, 0) * vector[vectorOffset]
                + signedByte(packed0, 32) * vector[vectorOffset + 4];
        sum0 += signedByte(packed1, 0) * vector[vectorOffset + 8]
                + signedByte(packed1, 32) * vector[vectorOffset + 12];
        sum0 += signedByte(packed2, 0) * vector[vectorOffset + 16]
                + signedByte(packed2, 32) * vector[vectorOffset + 20];
        sum0 += signedByte(packed3, 0) * vector[vectorOffset + 24]
                + signedByte(packed3, 32) * vector[vectorOffset + 28];
        float sum1 = signedByte(packed0, 8) * vector[vectorOffset + 1]
                + signedByte(packed0, 40) * vector[vectorOffset + 5];
        sum1 += signedByte(packed1, 8) * vector[vectorOffset + 9]
                + signedByte(packed1, 40) * vector[vectorOffset + 13];
        sum1 += signedByte(packed2, 8) * vector[vectorOffset + 17]
                + signedByte(packed2, 40) * vector[vectorOffset + 21];
        sum1 += signedByte(packed3, 8) * vector[vectorOffset + 25]
                + signedByte(packed3, 40) * vector[vectorOffset + 29];
        float sum2 = signedByte(packed0, 16) * vector[vectorOffset + 2]
                + signedByte(packed0, 48) * vector[vectorOffset + 6];
        sum2 += signedByte(packed1, 16) * vector[vectorOffset + 10]
                + signedByte(packed1, 48) * vector[vectorOffset + 14];
        sum2 += signedByte(packed2, 16) * vector[vectorOffset + 18]
                + signedByte(packed2, 48) * vector[vectorOffset + 22];
        sum2 += signedByte(packed3, 16) * vector[vectorOffset + 26]
                + signedByte(packed3, 48) * vector[vectorOffset + 30];
        float sum3 = signedByte(packed0, 24) * vector[vectorOffset + 3]
                + signedByte(packed0, 56) * vector[vectorOffset + 7];
        sum3 += signedByte(packed1, 24) * vector[vectorOffset + 11]
                + signedByte(packed1, 56) * vector[vectorOffset + 15];
        sum3 += signedByte(packed2, 24) * vector[vectorOffset + 19]
                + signedByte(packed2, 56) * vector[vectorOffset + 23];
        sum3 += signedByte(packed3, 24) * vector[vectorOffset + 27]
                + signedByte(packed3, 56) * vector[vectorOffset + 31];
        return sum0 + sum1 + sum2 + sum3;
    }

    static float dotRowQ8K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        if (RAW_Q8_VECTOR_DOT_ENABLED) {
            return dotRowQ8KVector(segment, rowOffset, columns, vector, vectorOffset);
        }
        return dotRowQ8KScalar(segment, rowOffset, columns, vector, vectorOffset);
    }

    static float dotRowQ8KVector(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        FloatVector acc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            acc0 = accumulateScaledQ8KBlockVector(
                    segment,
                    blockOffset + Float.BYTES,
                    vector,
                    vectorBase,
                    segment.get(LE_FLOAT, blockOffset),
                    acc0);
            acc1 = accumulateScaledQ8KBlockVector(
                    segment,
                    blockOffset + Q8_K_BLOCK_BYTES + Float.BYTES,
                    vector,
                    vectorBase + QK_K,
                    segment.get(LE_FLOAT, blockOffset + Q8_K_BLOCK_BYTES),
                    acc1);
            acc2 = accumulateScaledQ8KBlockVector(
                    segment,
                    blockOffset + 2L * Q8_K_BLOCK_BYTES + Float.BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    segment.get(LE_FLOAT, blockOffset + 2L * Q8_K_BLOCK_BYTES),
                    acc2);
            acc3 = accumulateScaledQ8KBlockVector(
                    segment,
                    blockOffset + 3L * Q8_K_BLOCK_BYTES + Float.BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    segment.get(LE_FLOAT, blockOffset + 3L * Q8_K_BLOCK_BYTES),
                    acc3);
            blockOffset += 4L * Q8_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
        for (; block < blocks; block++) {
            acc = accumulateScaledQ8KBlockVector(
                    segment,
                    blockOffset + Float.BYTES,
                    vector,
                    vectorBase,
                    segment.get(LE_FLOAT, blockOffset),
                    acc);
            blockOffset += Q8_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    static void dotRowsQ8KVector4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
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
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            for (int lane = 0; lane < 4; lane++) {
                long row0Block = row0Offset + blockOffset;
                long row1Block = row1Offset + blockOffset;
                long row2Block = row2Offset + blockOffset;
                long row3Block = row3Offset + blockOffset;
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

                for (int subBlock = 0; subBlock < 8; subBlock++) {
                    int subVectorBase = vectorBase + subBlock * Q8_0_BLOCK_SIZE;
                    long quantOffset = Float.BYTES + subBlock * (long) Q8_0_BLOCK_SIZE;
                    FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, subVectorBase);
                    FloatVector vf1 = FloatVector.fromArray(
                            Q4_DOT_FLOAT_SPECIES, vector, subVectorBase + Q4_DOT_VECTOR_LANES);
                    FloatVector vf2 = FloatVector.fromArray(
                            Q4_DOT_FLOAT_SPECIES, vector, subVectorBase + 2 * Q4_DOT_VECTOR_LANES);
                    FloatVector vf3 = FloatVector.fromArray(
                            Q4_DOT_FLOAT_SPECIES, vector, subVectorBase + 3 * Q4_DOT_VECTOR_LANES);
                    switch (subBlock & 3) {
                        case 0 -> {
                            row0BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc0);
                            row1BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc0);
                            row2BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc0);
                            row3BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc0);
                        }
                        case 1 -> {
                            row0BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc1);
                            row1BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc1);
                            row2BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc1);
                            row3BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc1);
                        }
                        case 2 -> {
                            row0BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc2);
                            row1BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc2);
                            row2BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc2);
                            row3BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc2);
                        }
                        default -> {
                            row0BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                    segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc3);
                            row1BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                    segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc3);
                            row2BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                    segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc3);
                            row3BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                    segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc3);
                        }
                    }
                }
                switch (lane) {
                    case 0 -> {
                        row0Acc0 = finishScaledQ8KBlockVector(
                                row0BlockAcc0, row0BlockAcc1, row0BlockAcc2, row0BlockAcc3,
                                segment.get(LE_FLOAT, row0Block), row0Acc0);
                        row1Acc0 = finishScaledQ8KBlockVector(
                                row1BlockAcc0, row1BlockAcc1, row1BlockAcc2, row1BlockAcc3,
                                segment.get(LE_FLOAT, row1Block), row1Acc0);
                        row2Acc0 = finishScaledQ8KBlockVector(
                                row2BlockAcc0, row2BlockAcc1, row2BlockAcc2, row2BlockAcc3,
                                segment.get(LE_FLOAT, row2Block), row2Acc0);
                        row3Acc0 = finishScaledQ8KBlockVector(
                                row3BlockAcc0, row3BlockAcc1, row3BlockAcc2, row3BlockAcc3,
                                segment.get(LE_FLOAT, row3Block), row3Acc0);
                    }
                    case 1 -> {
                        row0Acc1 = finishScaledQ8KBlockVector(
                                row0BlockAcc0, row0BlockAcc1, row0BlockAcc2, row0BlockAcc3,
                                segment.get(LE_FLOAT, row0Block), row0Acc1);
                        row1Acc1 = finishScaledQ8KBlockVector(
                                row1BlockAcc0, row1BlockAcc1, row1BlockAcc2, row1BlockAcc3,
                                segment.get(LE_FLOAT, row1Block), row1Acc1);
                        row2Acc1 = finishScaledQ8KBlockVector(
                                row2BlockAcc0, row2BlockAcc1, row2BlockAcc2, row2BlockAcc3,
                                segment.get(LE_FLOAT, row2Block), row2Acc1);
                        row3Acc1 = finishScaledQ8KBlockVector(
                                row3BlockAcc0, row3BlockAcc1, row3BlockAcc2, row3BlockAcc3,
                                segment.get(LE_FLOAT, row3Block), row3Acc1);
                    }
                    case 2 -> {
                        row0Acc2 = finishScaledQ8KBlockVector(
                                row0BlockAcc0, row0BlockAcc1, row0BlockAcc2, row0BlockAcc3,
                                segment.get(LE_FLOAT, row0Block), row0Acc2);
                        row1Acc2 = finishScaledQ8KBlockVector(
                                row1BlockAcc0, row1BlockAcc1, row1BlockAcc2, row1BlockAcc3,
                                segment.get(LE_FLOAT, row1Block), row1Acc2);
                        row2Acc2 = finishScaledQ8KBlockVector(
                                row2BlockAcc0, row2BlockAcc1, row2BlockAcc2, row2BlockAcc3,
                                segment.get(LE_FLOAT, row2Block), row2Acc2);
                        row3Acc2 = finishScaledQ8KBlockVector(
                                row3BlockAcc0, row3BlockAcc1, row3BlockAcc2, row3BlockAcc3,
                                segment.get(LE_FLOAT, row3Block), row3Acc2);
                    }
                    default -> {
                        row0Acc3 = finishScaledQ8KBlockVector(
                                row0BlockAcc0, row0BlockAcc1, row0BlockAcc2, row0BlockAcc3,
                                segment.get(LE_FLOAT, row0Block), row0Acc3);
                        row1Acc3 = finishScaledQ8KBlockVector(
                                row1BlockAcc0, row1BlockAcc1, row1BlockAcc2, row1BlockAcc3,
                                segment.get(LE_FLOAT, row1Block), row1Acc3);
                        row2Acc3 = finishScaledQ8KBlockVector(
                                row2BlockAcc0, row2BlockAcc1, row2BlockAcc2, row2BlockAcc3,
                                segment.get(LE_FLOAT, row2Block), row2Acc3);
                        row3Acc3 = finishScaledQ8KBlockVector(
                                row3BlockAcc0, row3BlockAcc1, row3BlockAcc2, row3BlockAcc3,
                                segment.get(LE_FLOAT, row3Block), row3Acc3);
                    }
                }
                blockOffset += Q8_K_BLOCK_BYTES;
                vectorBase += QK_K;
            }
        }
        FloatVector row0Acc = row0Acc0.add(row0Acc1).add(row0Acc2).add(row0Acc3);
        FloatVector row1Acc = row1Acc0.add(row1Acc1).add(row1Acc2).add(row1Acc3);
        FloatVector row2Acc = row2Acc0.add(row2Acc1).add(row2Acc2).add(row2Acc3);
        FloatVector row3Acc = row3Acc0.add(row3Acc1).add(row3Acc2).add(row3Acc3);
        for (; block < blocks; block++) {
            long row0Block = row0Offset + blockOffset;
            long row1Block = row1Offset + blockOffset;
            long row2Block = row2Offset + blockOffset;
            long row3Block = row3Offset + blockOffset;
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
            for (int subBlock = 0; subBlock < 8; subBlock++) {
                int subVectorBase = vectorBase + subBlock * Q8_0_BLOCK_SIZE;
                long quantOffset = Float.BYTES + subBlock * (long) Q8_0_BLOCK_SIZE;
                FloatVector vf0 = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, subVectorBase);
                FloatVector vf1 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, subVectorBase + Q4_DOT_VECTOR_LANES);
                FloatVector vf2 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, subVectorBase + 2 * Q4_DOT_VECTOR_LANES);
                FloatVector vf3 = FloatVector.fromArray(
                        Q4_DOT_FLOAT_SPECIES, vector, subVectorBase + 3 * Q4_DOT_VECTOR_LANES);
                switch (subBlock & 3) {
                    case 0 -> {
                        row0BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc0);
                        row1BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc0);
                        row2BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc0);
                        row3BlockAcc0 = accumulateQ8UnscaledBlockVector(
                                segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc0);
                    }
                    case 1 -> {
                        row0BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc1);
                        row1BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc1);
                        row2BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc1);
                        row3BlockAcc1 = accumulateQ8UnscaledBlockVector(
                                segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc1);
                    }
                    case 2 -> {
                        row0BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc2);
                        row1BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc2);
                        row2BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc2);
                        row3BlockAcc2 = accumulateQ8UnscaledBlockVector(
                                segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc2);
                    }
                    default -> {
                        row0BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                segment, row0Block + quantOffset, vf0, vf1, vf2, vf3, row0BlockAcc3);
                        row1BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                segment, row1Block + quantOffset, vf0, vf1, vf2, vf3, row1BlockAcc3);
                        row2BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                segment, row2Block + quantOffset, vf0, vf1, vf2, vf3, row2BlockAcc3);
                        row3BlockAcc3 = accumulateQ8UnscaledBlockVector(
                                segment, row3Block + quantOffset, vf0, vf1, vf2, vf3, row3BlockAcc3);
                    }
                }
            }
            row0Acc = finishScaledQ8KBlockVector(
                    row0BlockAcc0, row0BlockAcc1, row0BlockAcc2, row0BlockAcc3,
                    segment.get(LE_FLOAT, row0Block), row0Acc);
            row1Acc = finishScaledQ8KBlockVector(
                    row1BlockAcc0, row1BlockAcc1, row1BlockAcc2, row1BlockAcc3,
                    segment.get(LE_FLOAT, row1Block), row1Acc);
            row2Acc = finishScaledQ8KBlockVector(
                    row2BlockAcc0, row2BlockAcc1, row2BlockAcc2, row2BlockAcc3,
                    segment.get(LE_FLOAT, row2Block), row2Acc);
            row3Acc = finishScaledQ8KBlockVector(
                    row3BlockAcc0, row3BlockAcc1, row3BlockAcc2, row3BlockAcc3,
                    segment.get(LE_FLOAT, row3Block), row3Acc);
            blockOffset += Q8_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        output[outputOffset] = row0Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 1] = row1Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 2] = row2Acc.reduceLanes(VectorOperators.ADD);
        output[outputOffset + 3] = row3Acc.reduceLanes(VectorOperators.ADD);
    }

    static float dotRowQ8KScalar(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += segment.get(LE_FLOAT, blockOffset)
                    * dotRowQ8KBlockScalar(segment, blockOffset + Float.BYTES, vector, vectorBase);
            sum1 += segment.get(LE_FLOAT, blockOffset + Q8_K_BLOCK_BYTES)
                    * dotRowQ8KBlockScalar(
                            segment,
                            blockOffset + Q8_K_BLOCK_BYTES + Float.BYTES,
                            vector,
                            vectorBase + QK_K);
            sum2 += segment.get(LE_FLOAT, blockOffset + 2L * Q8_K_BLOCK_BYTES)
                    * dotRowQ8KBlockScalar(
                            segment,
                            blockOffset + 2L * Q8_K_BLOCK_BYTES + Float.BYTES,
                            vector,
                            vectorBase + 2 * QK_K);
            sum3 += segment.get(LE_FLOAT, blockOffset + 3L * Q8_K_BLOCK_BYTES)
                    * dotRowQ8KBlockScalar(
                            segment,
                            blockOffset + 3L * Q8_K_BLOCK_BYTES + Float.BYTES,
                            vector,
                            vectorBase + 3 * QK_K);
            blockOffset += 4L * Q8_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            float d = segment.get(LE_FLOAT, blockOffset);
            float quantDot = dotRowQ8KBlockScalar(segment, blockOffset + Float.BYTES, vector, vectorBase);
            sum += d * quantDot;
            blockOffset += Q8_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return sum;
    }

    static void dotRowsQ8KScalar4(
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
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotScaledQ8KScalarBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotScaledQ8KScalarBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotScaledQ8KScalarBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotScaledQ8KScalarBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotScaledQ8KScalarBlock(
                    segment,
                    row0Offset + blockOffset + Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row1Sum1 += dotScaledQ8KScalarBlock(
                    segment,
                    row1Offset + blockOffset + Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row2Sum1 += dotScaledQ8KScalarBlock(
                    segment,
                    row2Offset + blockOffset + Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row3Sum1 += dotScaledQ8KScalarBlock(
                    segment,
                    row3Offset + blockOffset + Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row0Sum2 += dotScaledQ8KScalarBlock(
                    segment,
                    row0Offset + blockOffset + 2L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row1Sum2 += dotScaledQ8KScalarBlock(
                    segment,
                    row1Offset + blockOffset + 2L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row2Sum2 += dotScaledQ8KScalarBlock(
                    segment,
                    row2Offset + blockOffset + 2L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row3Sum2 += dotScaledQ8KScalarBlock(
                    segment,
                    row3Offset + blockOffset + 2L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row0Sum3 += dotScaledQ8KScalarBlock(
                    segment,
                    row0Offset + blockOffset + 3L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            row1Sum3 += dotScaledQ8KScalarBlock(
                    segment,
                    row1Offset + blockOffset + 3L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            row2Sum3 += dotScaledQ8KScalarBlock(
                    segment,
                    row2Offset + blockOffset + 3L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            row3Sum3 += dotScaledQ8KScalarBlock(
                    segment,
                    row3Offset + blockOffset + 3L * Q8_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            blockOffset += 4L * Q8_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotScaledQ8KScalarBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotScaledQ8KScalarBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotScaledQ8KScalarBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotScaledQ8KScalarBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += Q8_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotScaledQ8KScalarBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = segment.get(LE_FLOAT, blockOffset);
        return d * dotRowQ8KBlockScalar(segment, blockOffset + Float.BYTES, vector, vectorBase);
    }

    private static FloatVector accumulateScaledQ8KBlockVector(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset,
            float scale,
            FloatVector acc) {
        FloatVector blockAcc0 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector blockAcc1 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector blockAcc2 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        FloatVector blockAcc3 = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        blockAcc0 = accumulateQ8UnscaledBlockVector(segment, quantsOffset, vector, vectorOffset, blockAcc0);
        blockAcc1 = accumulateQ8UnscaledBlockVector(
                segment,
                quantsOffset + Q8_0_BLOCK_SIZE,
                vector,
                vectorOffset + Q8_0_BLOCK_SIZE,
                blockAcc1);
        blockAcc2 = accumulateQ8UnscaledBlockVector(
                segment,
                quantsOffset + 2L * Q8_0_BLOCK_SIZE,
                vector,
                vectorOffset + 2 * Q8_0_BLOCK_SIZE,
                blockAcc2);
        blockAcc3 = accumulateQ8UnscaledBlockVector(
                segment,
                quantsOffset + 3L * Q8_0_BLOCK_SIZE,
                vector,
                vectorOffset + 3 * Q8_0_BLOCK_SIZE,
                blockAcc3);
        blockAcc0 = accumulateQ8UnscaledBlockVector(
                segment,
                quantsOffset + 4L * Q8_0_BLOCK_SIZE,
                vector,
                vectorOffset + 4 * Q8_0_BLOCK_SIZE,
                blockAcc0);
        blockAcc1 = accumulateQ8UnscaledBlockVector(
                segment,
                quantsOffset + 5L * Q8_0_BLOCK_SIZE,
                vector,
                vectorOffset + 5 * Q8_0_BLOCK_SIZE,
                blockAcc1);
        blockAcc2 = accumulateQ8UnscaledBlockVector(
                segment,
                quantsOffset + 6L * Q8_0_BLOCK_SIZE,
                vector,
                vectorOffset + 6 * Q8_0_BLOCK_SIZE,
                blockAcc2);
        blockAcc3 = accumulateQ8UnscaledBlockVector(
                segment,
                quantsOffset + 7L * Q8_0_BLOCK_SIZE,
                vector,
                vectorOffset + 7 * Q8_0_BLOCK_SIZE,
                blockAcc3);
        return finishScaledQ8KBlockVector(blockAcc0, blockAcc1, blockAcc2, blockAcc3, scale, acc);
    }

    private static FloatVector finishScaledQ8KBlockVector(
            FloatVector blockAcc0,
            FloatVector blockAcc1,
            FloatVector blockAcc2,
            FloatVector blockAcc3,
            float scale,
            FloatVector acc) {
        FloatVector blockAcc = blockAcc0.add(blockAcc1).add(blockAcc2).add(blockAcc3);
        return blockAcc.fma(FloatVector.broadcast(Q4_DOT_FLOAT_SPECIES, scale), acc);
    }

    private static float dotRowQ8KBlockScalar(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockQuants = quantsOffset;
        int blockVector = vectorOffset;
        for (int block = 0; block < 8; block++) {
            long packed0 = segment.get(LE_LONG, blockQuants);
            long packed1 = segment.get(LE_LONG, blockQuants + Long.BYTES);
            long packed2 = segment.get(LE_LONG, blockQuants + 2L * Long.BYTES);
            long packed3 = segment.get(LE_LONG, blockQuants + 3L * Long.BYTES);
            sum0 += signedByte(packed0, 0) * vector[blockVector]
                    + signedByte(packed0, 32) * vector[blockVector + 4];
            sum0 += signedByte(packed1, 0) * vector[blockVector + 8]
                    + signedByte(packed1, 32) * vector[blockVector + 12];
            sum0 += signedByte(packed2, 0) * vector[blockVector + 16]
                    + signedByte(packed2, 32) * vector[blockVector + 20];
            sum0 += signedByte(packed3, 0) * vector[blockVector + 24]
                    + signedByte(packed3, 32) * vector[blockVector + 28];
            sum1 += signedByte(packed0, 8) * vector[blockVector + 1]
                    + signedByte(packed0, 40) * vector[blockVector + 5];
            sum1 += signedByte(packed1, 8) * vector[blockVector + 9]
                    + signedByte(packed1, 40) * vector[blockVector + 13];
            sum1 += signedByte(packed2, 8) * vector[blockVector + 17]
                    + signedByte(packed2, 40) * vector[blockVector + 21];
            sum1 += signedByte(packed3, 8) * vector[blockVector + 25]
                    + signedByte(packed3, 40) * vector[blockVector + 29];
            sum2 += signedByte(packed0, 16) * vector[blockVector + 2]
                    + signedByte(packed0, 48) * vector[blockVector + 6];
            sum2 += signedByte(packed1, 16) * vector[blockVector + 10]
                    + signedByte(packed1, 48) * vector[blockVector + 14];
            sum2 += signedByte(packed2, 16) * vector[blockVector + 18]
                    + signedByte(packed2, 48) * vector[blockVector + 22];
            sum2 += signedByte(packed3, 16) * vector[blockVector + 26]
                    + signedByte(packed3, 48) * vector[blockVector + 30];
            sum3 += signedByte(packed0, 24) * vector[blockVector + 3]
                    + signedByte(packed0, 56) * vector[blockVector + 7];
            sum3 += signedByte(packed1, 24) * vector[blockVector + 11]
                    + signedByte(packed1, 56) * vector[blockVector + 15];
            sum3 += signedByte(packed2, 24) * vector[blockVector + 19]
                    + signedByte(packed2, 56) * vector[blockVector + 23];
            sum3 += signedByte(packed3, 24) * vector[blockVector + 27]
                    + signedByte(packed3, 56) * vector[blockVector + 31];
            blockQuants += Q8_0_BLOCK_SIZE;
            blockVector += Q8_0_BLOCK_SIZE;
        }
        return sum0 + sum1 + sum2 + sum3;
    }
}
