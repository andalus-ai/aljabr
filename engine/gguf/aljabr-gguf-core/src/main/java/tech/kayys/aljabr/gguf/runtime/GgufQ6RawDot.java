package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.signedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q6_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_GROUPS_PER_BLOCK;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_GROUPS_PER_SUPER_BLOCK;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw Q6_K row-dot kernel.
 *
 * <p>Q6_K combines four-bit low nibbles, two high-bit planes, and signed group
 * scales. Isolating the decode keeps the public tensor facade out of the
 * bit-packing details and gives future vector or Metal parity work a narrow
 * target.</p>
 * @author bhangun
 */
final class GgufQ6RawDot {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ6RawDot() {
    }

    static float dotRowQ6K(
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
            sum0 += dotRowQ6KBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ6KBlock(
                    segment,
                    blockOffset + Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            sum2 += dotRowQ6KBlock(
                    segment,
                    blockOffset + 2L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            sum3 += dotRowQ6KBlock(
                    segment,
                    blockOffset + 3L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            blockOffset += 4L * Q6_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ6KBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += Q6_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return sum;
    }

    private static float dotRowQ6KBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 208));
        long lowBitsBase = blockOffset;
        long highBitsBase = blockOffset + 128;
        long scalesOffset = blockOffset + 192;
        long firstScales = segment.get(LE_LONG, scalesOffset);
        float sum0 = dotRowQ6KHalf(
                segment,
                lowBitsBase,
                highBitsBase,
                vector,
                vectorBase,
                0,
                d * signedByte(firstScales, 0),
                d * signedByte(firstScales, 16),
                d * signedByte(firstScales, 32),
                d * signedByte(firstScales, 48));
        float sum1 = dotRowQ6KHalf(
                segment,
                lowBitsBase,
                highBitsBase,
                vector,
                vectorBase,
                16,
                d * signedByte(firstScales, 8),
                d * signedByte(firstScales, 24),
                d * signedByte(firstScales, 40),
                d * signedByte(firstScales, 56));
        long secondScales = segment.get(LE_LONG, scalesOffset + 8);
        float sum2 = dotRowQ6KHalf(
                segment,
                lowBitsBase + 64,
                highBitsBase + 32,
                vector,
                vectorBase + 128,
                0,
                d * signedByte(secondScales, 0),
                d * signedByte(secondScales, 16),
                d * signedByte(secondScales, 32),
                d * signedByte(secondScales, 48));
        float sum3 = dotRowQ6KHalf(
                segment,
                lowBitsBase + 64,
                highBitsBase + 32,
                vector,
                vectorBase + 128,
                16,
                d * signedByte(secondScales, 8),
                d * signedByte(secondScales, 24),
                d * signedByte(secondScales, 40),
                d * signedByte(secondScales, 56));
        return sum0 + sum1 + sum2 + sum3;
    }

    static float dotRowQ6KWithGroupSums(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = 0;
        int groupBase = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ6KGroupSumBlock(segment, blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            sum1 += dotRowQ6KGroupSumBlock(
                    segment,
                    blockOffset + Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupBase + QK_GROUPS_PER_BLOCK);
            sum2 += dotRowQ6KGroupSumBlock(
                    segment,
                    blockOffset + 2L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupBase + 2 * QK_GROUPS_PER_BLOCK);
            sum3 += dotRowQ6KGroupSumBlock(
                    segment,
                    blockOffset + 3L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupBase + 3 * QK_GROUPS_PER_BLOCK);
            blockOffset += 4L * Q6_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
            groupBase += 4 * QK_GROUPS_PER_BLOCK;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ6KGroupSumBlock(segment, blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            blockOffset += Q6_K_BLOCK_BYTES;
            vectorBase += QK_K;
            groupBase += QK_GROUPS_PER_BLOCK;
        }
        return sum;
    }

    static void dotRowsQ6KWithGroupSums4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
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
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int groupBase = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ6KGroupSumBlock(segment, row0Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            row1Sum0 += dotRowQ6KGroupSumBlock(segment, row1Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            row2Sum0 += dotRowQ6KGroupSumBlock(segment, row2Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            row3Sum0 += dotRowQ6KGroupSumBlock(segment, row3Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            row0Sum1 += dotRowQ6KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupBase + QK_GROUPS_PER_BLOCK);
            row1Sum1 += dotRowQ6KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupBase + QK_GROUPS_PER_BLOCK);
            row2Sum1 += dotRowQ6KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupBase + QK_GROUPS_PER_BLOCK);
            row3Sum1 += dotRowQ6KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupBase + QK_GROUPS_PER_BLOCK);
            row0Sum2 += dotRowQ6KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + 2L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupBase + 2 * QK_GROUPS_PER_BLOCK);
            row1Sum2 += dotRowQ6KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + 2L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupBase + 2 * QK_GROUPS_PER_BLOCK);
            row2Sum2 += dotRowQ6KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + 2L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupBase + 2 * QK_GROUPS_PER_BLOCK);
            row3Sum2 += dotRowQ6KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + 2L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupBase + 2 * QK_GROUPS_PER_BLOCK);
            row0Sum3 += dotRowQ6KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + 3L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupBase + 3 * QK_GROUPS_PER_BLOCK);
            row1Sum3 += dotRowQ6KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + 3L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupBase + 3 * QK_GROUPS_PER_BLOCK);
            row2Sum3 += dotRowQ6KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + 3L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupBase + 3 * QK_GROUPS_PER_BLOCK);
            row3Sum3 += dotRowQ6KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + 3L * Q6_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupBase + 3 * QK_GROUPS_PER_BLOCK);
            blockOffset += 4L * Q6_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
            groupBase += 4 * QK_GROUPS_PER_BLOCK;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ6KGroupSumBlock(segment, row0Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            row1Sum += dotRowQ6KGroupSumBlock(segment, row1Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            row2Sum += dotRowQ6KGroupSumBlock(segment, row2Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            row3Sum += dotRowQ6KGroupSumBlock(segment, row3Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupBase);
            blockOffset += Q6_K_BLOCK_BYTES;
            vectorBase += QK_K;
            groupBase += QK_GROUPS_PER_BLOCK;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowQ6KGroupSumBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase,
            float[] vectorGroupSums,
            int groupBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 208));
        long lowBitsBase = blockOffset;
        long highBitsBase = blockOffset + 128;
        long scalesOffset = blockOffset + 192;
        long firstScales = segment.get(LE_LONG, scalesOffset);
        float sum0 = dotRowQ6KHalfWithGroupSums(
                segment,
                lowBitsBase,
                highBitsBase,
                vector,
                vectorBase,
                0,
                d * signedByte(firstScales, 0),
                d * signedByte(firstScales, 16),
                d * signedByte(firstScales, 32),
                d * signedByte(firstScales, 48),
                vectorGroupSums,
                groupBase);
        float sum1 = dotRowQ6KHalfWithGroupSums(
                segment,
                lowBitsBase,
                highBitsBase,
                vector,
                vectorBase,
                16,
                d * signedByte(firstScales, 8),
                d * signedByte(firstScales, 24),
                d * signedByte(firstScales, 40),
                d * signedByte(firstScales, 56),
                vectorGroupSums,
                groupBase + 1);
        long secondScales = segment.get(LE_LONG, scalesOffset + 8);
        float sum2 = dotRowQ6KHalfWithGroupSums(
                segment,
                lowBitsBase + 64,
                highBitsBase + 32,
                vector,
                vectorBase + 128,
                0,
                d * signedByte(secondScales, 0),
                d * signedByte(secondScales, 16),
                d * signedByte(secondScales, 32),
                d * signedByte(secondScales, 48),
                vectorGroupSums,
                groupBase + QK_GROUPS_PER_SUPER_BLOCK);
        float sum3 = dotRowQ6KHalfWithGroupSums(
                segment,
                lowBitsBase + 64,
                highBitsBase + 32,
                vector,
                vectorBase + 128,
                16,
                d * signedByte(secondScales, 8),
                d * signedByte(secondScales, 24),
                d * signedByte(secondScales, 40),
                d * signedByte(secondScales, 56),
                vectorGroupSums,
                groupBase + QK_GROUPS_PER_SUPER_BLOCK + 1);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotRowQ6KHalf(
            MemorySegment segment,
            long lowBitsBase,
            long highBitsBase,
            float[] vector,
            int vectorBase,
            int halfOffset,
            float scale1,
            float scale2,
            float scale3,
            float scale4) {
        float quantDot1 = 0.0f;
        float quantDot2 = 0.0f;
        float quantDot3 = 0.0f;
        float quantDot4 = 0.0f;
        int halfEnd = halfOffset + 16;
        for (int i = halfOffset; i < halfEnd; i += Long.BYTES) {
            long packedLowA = segment.get(LE_LONG, lowBitsBase + i);
            long packedLowB = segment.get(LE_LONG, lowBitsBase + 32 + i);
            long packedHighBits = segment.get(LE_LONG, highBitsBase + i);
            int lowA0 = unsignedByte(packedLowA, 0);
            int lowA1 = unsignedByte(packedLowA, 8);
            int lowA2 = unsignedByte(packedLowA, 16);
            int lowA3 = unsignedByte(packedLowA, 24);
            int lowA4 = unsignedByte(packedLowA, 32);
            int lowA5 = unsignedByte(packedLowA, 40);
            int lowA6 = unsignedByte(packedLowA, 48);
            int lowA7 = unsignedByte(packedLowA, 56);
            int lowB0 = unsignedByte(packedLowB, 0);
            int lowB1 = unsignedByte(packedLowB, 8);
            int lowB2 = unsignedByte(packedLowB, 16);
            int lowB3 = unsignedByte(packedLowB, 24);
            int lowB4 = unsignedByte(packedLowB, 32);
            int lowB5 = unsignedByte(packedLowB, 40);
            int lowB6 = unsignedByte(packedLowB, 48);
            int lowB7 = unsignedByte(packedLowB, 56);
            int highBits0 = unsignedByte(packedHighBits, 0);
            int highBits1 = unsignedByte(packedHighBits, 8);
            int highBits2 = unsignedByte(packedHighBits, 16);
            int highBits3 = unsignedByte(packedHighBits, 24);
            int highBits4 = unsignedByte(packedHighBits, 32);
            int highBits5 = unsignedByte(packedHighBits, 40);
            int highBits6 = unsignedByte(packedHighBits, 48);
            int highBits7 = unsignedByte(packedHighBits, 56);
            quantDot1 += (((lowA0 & 0x0F) | (((highBits0 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i]
                    + (((lowA1 & 0x0F) | (((highBits1 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i + 1]
                    + (((lowA2 & 0x0F) | (((highBits2 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i + 2]
                    + (((lowA3 & 0x0F) | (((highBits3 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i + 3]
                    + (((lowA4 & 0x0F) | (((highBits4 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i + 4]
                    + (((lowA5 & 0x0F) | (((highBits5 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i + 5]
                    + (((lowA6 & 0x0F) | (((highBits6 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i + 6]
                    + (((lowA7 & 0x0F) | (((highBits7 >>> 0) & 0x03) << 4)) - 32)
                            * vector[vectorBase + i + 7];
            quantDot2 += (((lowB0 & 0x0F) | (((highBits0 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i]
                    + (((lowB1 & 0x0F) | (((highBits1 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i + 1]
                    + (((lowB2 & 0x0F) | (((highBits2 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i + 2]
                    + (((lowB3 & 0x0F) | (((highBits3 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i + 3]
                    + (((lowB4 & 0x0F) | (((highBits4 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i + 4]
                    + (((lowB5 & 0x0F) | (((highBits5 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i + 5]
                    + (((lowB6 & 0x0F) | (((highBits6 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i + 6]
                    + (((lowB7 & 0x0F) | (((highBits7 >>> 2) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 32 + i + 7];
            quantDot3 += (((lowA0 >>> 4) | (((highBits0 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i]
                    + (((lowA1 >>> 4) | (((highBits1 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i + 1]
                    + (((lowA2 >>> 4) | (((highBits2 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i + 2]
                    + (((lowA3 >>> 4) | (((highBits3 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i + 3]
                    + (((lowA4 >>> 4) | (((highBits4 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i + 4]
                    + (((lowA5 >>> 4) | (((highBits5 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i + 5]
                    + (((lowA6 >>> 4) | (((highBits6 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i + 6]
                    + (((lowA7 >>> 4) | (((highBits7 >>> 4) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 64 + i + 7];
            quantDot4 += (((lowB0 >>> 4) | (((highBits0 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i]
                    + (((lowB1 >>> 4) | (((highBits1 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i + 1]
                    + (((lowB2 >>> 4) | (((highBits2 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i + 2]
                    + (((lowB3 >>> 4) | (((highBits3 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i + 3]
                    + (((lowB4 >>> 4) | (((highBits4 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i + 4]
                    + (((lowB5 >>> 4) | (((highBits5 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i + 5]
                    + (((lowB6 >>> 4) | (((highBits6 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i + 6]
                    + (((lowB7 >>> 4) | (((highBits7 >>> 6) & 0x03) << 4)) - 32)
                            * vector[vectorBase + 96 + i + 7];
        }
        return scale1 * quantDot1 + scale2 * quantDot2 + scale3 * quantDot3 + scale4 * quantDot4;
    }

    private static float dotRowQ6KHalfWithGroupSums(
            MemorySegment segment,
            long lowBitsBase,
            long highBitsBase,
            float[] vector,
            int vectorBase,
            int halfOffset,
            float scale1,
            float scale2,
            float scale3,
            float scale4,
            float[] vectorGroupSums,
            int groupBase) {
        float quantDot1 = 0.0f;
        float quantDot2 = 0.0f;
        float quantDot3 = 0.0f;
        float quantDot4 = 0.0f;
        int halfEnd = halfOffset + 16;
        for (int i = halfOffset; i < halfEnd; i += Long.BYTES) {
            long packedLowA = segment.get(LE_LONG, lowBitsBase + i);
            long packedLowB = segment.get(LE_LONG, lowBitsBase + 32 + i);
            long packedHighBits = segment.get(LE_LONG, highBitsBase + i);
            int lowA0 = unsignedByte(packedLowA, 0);
            int lowA1 = unsignedByte(packedLowA, 8);
            int lowA2 = unsignedByte(packedLowA, 16);
            int lowA3 = unsignedByte(packedLowA, 24);
            int lowA4 = unsignedByte(packedLowA, 32);
            int lowA5 = unsignedByte(packedLowA, 40);
            int lowA6 = unsignedByte(packedLowA, 48);
            int lowA7 = unsignedByte(packedLowA, 56);
            int lowB0 = unsignedByte(packedLowB, 0);
            int lowB1 = unsignedByte(packedLowB, 8);
            int lowB2 = unsignedByte(packedLowB, 16);
            int lowB3 = unsignedByte(packedLowB, 24);
            int lowB4 = unsignedByte(packedLowB, 32);
            int lowB5 = unsignedByte(packedLowB, 40);
            int lowB6 = unsignedByte(packedLowB, 48);
            int lowB7 = unsignedByte(packedLowB, 56);
            int highBits0 = unsignedByte(packedHighBits, 0);
            int highBits1 = unsignedByte(packedHighBits, 8);
            int highBits2 = unsignedByte(packedHighBits, 16);
            int highBits3 = unsignedByte(packedHighBits, 24);
            int highBits4 = unsignedByte(packedHighBits, 32);
            int highBits5 = unsignedByte(packedHighBits, 40);
            int highBits6 = unsignedByte(packedHighBits, 48);
            int highBits7 = unsignedByte(packedHighBits, 56);
            quantDot1 += ((lowA0 & 0x0F) | (((highBits0 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i]
                    + ((lowA1 & 0x0F) | (((highBits1 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i + 1]
                    + ((lowA2 & 0x0F) | (((highBits2 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i + 2]
                    + ((lowA3 & 0x0F) | (((highBits3 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i + 3]
                    + ((lowA4 & 0x0F) | (((highBits4 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i + 4]
                    + ((lowA5 & 0x0F) | (((highBits5 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i + 5]
                    + ((lowA6 & 0x0F) | (((highBits6 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i + 6]
                    + ((lowA7 & 0x0F) | (((highBits7 >>> 0) & 0x03) << 4))
                            * vector[vectorBase + i + 7];
            quantDot2 += ((lowB0 & 0x0F) | (((highBits0 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i]
                    + ((lowB1 & 0x0F) | (((highBits1 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i + 1]
                    + ((lowB2 & 0x0F) | (((highBits2 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i + 2]
                    + ((lowB3 & 0x0F) | (((highBits3 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i + 3]
                    + ((lowB4 & 0x0F) | (((highBits4 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i + 4]
                    + ((lowB5 & 0x0F) | (((highBits5 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i + 5]
                    + ((lowB6 & 0x0F) | (((highBits6 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i + 6]
                    + ((lowB7 & 0x0F) | (((highBits7 >>> 2) & 0x03) << 4))
                            * vector[vectorBase + 32 + i + 7];
            quantDot3 += ((lowA0 >>> 4) | (((highBits0 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i]
                    + ((lowA1 >>> 4) | (((highBits1 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i + 1]
                    + ((lowA2 >>> 4) | (((highBits2 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i + 2]
                    + ((lowA3 >>> 4) | (((highBits3 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i + 3]
                    + ((lowA4 >>> 4) | (((highBits4 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i + 4]
                    + ((lowA5 >>> 4) | (((highBits5 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i + 5]
                    + ((lowA6 >>> 4) | (((highBits6 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i + 6]
                    + ((lowA7 >>> 4) | (((highBits7 >>> 4) & 0x03) << 4))
                            * vector[vectorBase + 64 + i + 7];
            quantDot4 += ((lowB0 >>> 4) | (((highBits0 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i]
                    + ((lowB1 >>> 4) | (((highBits1 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i + 1]
                    + ((lowB2 >>> 4) | (((highBits2 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i + 2]
                    + ((lowB3 >>> 4) | (((highBits3 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i + 3]
                    + ((lowB4 >>> 4) | (((highBits4 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i + 4]
                    + ((lowB5 >>> 4) | (((highBits5 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i + 5]
                    + ((lowB6 >>> 4) | (((highBits6 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i + 6]
                    + ((lowB7 >>> 4) | (((highBits7 >>> 6) & 0x03) << 4))
                            * vector[vectorBase + 96 + i + 7];
        }
        return scale1 * (quantDot1 - 32.0f * vectorGroupSums[groupBase])
                + scale2 * (quantDot2 - 32.0f * vectorGroupSums[groupBase + 2])
                + scale3 * (quantDot3 - 32.0f * vectorGroupSums[groupBase + 4])
                + scale4 * (quantDot4 - 32.0f * vectorGroupSums[groupBase + 6]);
    }
}
