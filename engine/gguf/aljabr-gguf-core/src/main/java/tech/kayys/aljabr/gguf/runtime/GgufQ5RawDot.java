package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.minContribution;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.minFromPackedScaleMin;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.scaleFromPackedScaleMin;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.scaleK4Packed;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.scaleMinK4PackedCode;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw Q5_K row-dot kernels.
 *
 * <p>Q5_K extends Q4_K's nibbles with separate high-bit planes. This helper
 * owns direct dot products, no-min fast paths, and the mat-vec variant that
 * reuses precomputed vector group sums for min correction.</p>
 * @author bhangun
 */
final class GgufQ5RawDot {
    private static final int QK_GROUPS = QK_K / 32;

    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ5RawDot() {
    }

    static float dotRowQ5K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / QK_K;
        for (int block = 0; block < blocks; block++) {
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
            boolean hasBlockMins = dMin != 0.0f;
            long scalesOffset = blockOffset + 4;
            long scalesLow = segment.get(LE_LONG, scalesOffset);
            int scalesHigh = segment.get(LE_INT, scalesOffset + Long.BYTES);
            long highBitsOffset = blockOffset + 16;
            long quantsOffset = blockOffset + 48;
            int scaleIndex = 0;
            int highMaskLow = 1;
            int highMaskHigh = 2;
            int highShiftLow = 0;
            int highShiftHigh = 1;
            int vBase = vectorBase;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
                int firstScaleIndex = scaleIndex++;
                int secondScaleIndex = scaleIndex++;
                float d1;
                float d2;
                int min1 = 0;
                int min2 = 0;
                if (!hasBlockMins) {
                    d1 = d * scaleK4Packed(scalesLow, scalesHigh, firstScaleIndex);
                    d2 = d * scaleK4Packed(scalesLow, scalesHigh, secondScaleIndex);
                } else {
                    int first = scaleMinK4PackedCode(scalesLow, scalesHigh, firstScaleIndex);
                    int second = scaleMinK4PackedCode(scalesLow, scalesHigh, secondScaleIndex);
                    d1 = d * scaleFromPackedScaleMin(first);
                    d2 = d * scaleFromPackedScaleMin(second);
                    min1 = minFromPackedScaleMin(first);
                    min2 = minFromPackedScaleMin(second);
                }
                boolean hasMin1 = hasBlockMins && min1 != 0;
                boolean hasMin2 = hasBlockMins && min2 != 0;

                float vectorSum1 = 0.0f;
                float vectorSum2 = 0.0f;
                float quantDot1 = 0.0f;
                float quantDot2 = 0.0f;
                for (int i = 0; i < 32; i += Long.BYTES) {
                    long packedQuants = segment.get(LE_LONG, quantsOffset + i);
                    long packedHighBits = segment.get(LE_LONG, highBitsOffset + i);
                    int quant0 = unsignedByte(packedQuants, 0);
                    int quant1 = unsignedByte(packedQuants, 8);
                    int quant2 = unsignedByte(packedQuants, 16);
                    int quant3 = unsignedByte(packedQuants, 24);
                    int quant4 = unsignedByte(packedQuants, 32);
                    int quant5 = unsignedByte(packedQuants, 40);
                    int quant6 = unsignedByte(packedQuants, 48);
                    int quant7 = unsignedByte(packedQuants, 56);
                    int highBits0 = unsignedByte(packedHighBits, 0);
                    int highBits1 = unsignedByte(packedHighBits, 8);
                    int highBits2 = unsignedByte(packedHighBits, 16);
                    int highBits3 = unsignedByte(packedHighBits, 24);
                    int highBits4 = unsignedByte(packedHighBits, 32);
                    int highBits5 = unsignedByte(packedHighBits, 40);
                    int highBits6 = unsignedByte(packedHighBits, 48);
                    int highBits7 = unsignedByte(packedHighBits, 56);
                    float v10 = vector[vBase + i];
                    float v11 = vector[vBase + i + 1];
                    float v12 = vector[vBase + i + 2];
                    float v13 = vector[vBase + i + 3];
                    float v14 = vector[vBase + i + 4];
                    float v15 = vector[vBase + i + 5];
                    float v16 = vector[vBase + i + 6];
                    float v17 = vector[vBase + i + 7];
                    float v20 = vector[vBase + 32 + i];
                    float v21 = vector[vBase + 32 + i + 1];
                    float v22 = vector[vBase + 32 + i + 2];
                    float v23 = vector[vBase + 32 + i + 3];
                    float v24 = vector[vBase + 32 + i + 4];
                    float v25 = vector[vBase + 32 + i + 5];
                    float v26 = vector[vBase + 32 + i + 6];
                    float v27 = vector[vBase + 32 + i + 7];
                    quantDot1 += ((quant0 & 0x0F) + (((highBits0 & highMaskLow) >>> highShiftLow) << 4)) * v10
                            + ((quant1 & 0x0F) + (((highBits1 & highMaskLow) >>> highShiftLow) << 4)) * v11
                            + ((quant2 & 0x0F) + (((highBits2 & highMaskLow) >>> highShiftLow) << 4)) * v12
                            + ((quant3 & 0x0F) + (((highBits3 & highMaskLow) >>> highShiftLow) << 4)) * v13
                            + ((quant4 & 0x0F) + (((highBits4 & highMaskLow) >>> highShiftLow) << 4)) * v14
                            + ((quant5 & 0x0F) + (((highBits5 & highMaskLow) >>> highShiftLow) << 4)) * v15
                            + ((quant6 & 0x0F) + (((highBits6 & highMaskLow) >>> highShiftLow) << 4)) * v16
                            + ((quant7 & 0x0F) + (((highBits7 & highMaskLow) >>> highShiftLow) << 4)) * v17;
                    quantDot2 += ((quant0 >>> 4) + (((highBits0 & highMaskHigh) >>> highShiftHigh) << 4)) * v20
                            + ((quant1 >>> 4) + (((highBits1 & highMaskHigh) >>> highShiftHigh) << 4)) * v21
                            + ((quant2 >>> 4) + (((highBits2 & highMaskHigh) >>> highShiftHigh) << 4)) * v22
                            + ((quant3 >>> 4) + (((highBits3 & highMaskHigh) >>> highShiftHigh) << 4)) * v23
                            + ((quant4 >>> 4) + (((highBits4 & highMaskHigh) >>> highShiftHigh) << 4)) * v24
                            + ((quant5 >>> 4) + (((highBits5 & highMaskHigh) >>> highShiftHigh) << 4)) * v25
                            + ((quant6 >>> 4) + (((highBits6 & highMaskHigh) >>> highShiftHigh) << 4)) * v26
                            + ((quant7 >>> 4) + (((highBits7 & highMaskHigh) >>> highShiftHigh) << 4)) * v27;
                    if (hasMin1) {
                        vectorSum1 += v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17;
                    }
                    if (hasMin2) {
                        vectorSum2 += v20 + v21 + v22 + v23 + v24 + v25 + v26 + v27;
                    }
                }
                sum += d1 * quantDot1 + d2 * quantDot2;
                if (hasMin1) {
                    sum -= dMin * min1 * vectorSum1;
                }
                if (hasMin2) {
                    sum -= dMin * min2 * vectorSum2;
                }
                quantsOffset += 32;
                highMaskLow <<= 2;
                highMaskHigh <<= 2;
                highShiftLow += 2;
                highShiftHigh += 2;
                vBase += 64;
            }
            blockOffset += Q5_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return sum;
    }

    static float dotRowQ5KNoMins(
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
            sum0 += dotRowQ5KNoMinBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ5KNoMinBlock(segment, blockOffset + Q5_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            sum2 += dotRowQ5KNoMinBlock(segment, blockOffset + 2L * Q5_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            sum3 += dotRowQ5KNoMinBlock(segment, blockOffset + 3L * Q5_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            blockOffset += 4L * Q5_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ5KNoMinBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += Q5_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return sum;
    }

    static void dotRowsQ5KNoMins4(
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
            row0Sum0 += dotRowQ5KNoMinBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowQ5KNoMinBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowQ5KNoMinBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowQ5KNoMinBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowQ5KNoMinBlock(segment, row0Offset + blockOffset + Q5_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row1Sum1 += dotRowQ5KNoMinBlock(segment, row1Offset + blockOffset + Q5_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row2Sum1 += dotRowQ5KNoMinBlock(segment, row2Offset + blockOffset + Q5_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row3Sum1 += dotRowQ5KNoMinBlock(segment, row3Offset + blockOffset + Q5_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row0Sum2 += dotRowQ5KNoMinBlock(segment, row0Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row1Sum2 += dotRowQ5KNoMinBlock(segment, row1Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row2Sum2 += dotRowQ5KNoMinBlock(segment, row2Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row3Sum2 += dotRowQ5KNoMinBlock(segment, row3Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row0Sum3 += dotRowQ5KNoMinBlock(segment, row0Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            row1Sum3 += dotRowQ5KNoMinBlock(segment, row1Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            row2Sum3 += dotRowQ5KNoMinBlock(segment, row2Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            row3Sum3 += dotRowQ5KNoMinBlock(segment, row3Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            blockOffset += 4L * Q5_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ5KNoMinBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowQ5KNoMinBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowQ5KNoMinBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowQ5KNoMinBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += Q5_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowQ5KNoMinBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        long scalesOffset = blockOffset + 4;
        long scalesLow = segment.get(LE_LONG, scalesOffset);
        int scalesHigh = segment.get(LE_INT, scalesOffset + Long.BYTES);
        long highBitsOffset = blockOffset + 16;
        long quantsOffset = blockOffset + 48;
        float sum0 = dotRowQ5KNoMinSuperBlock(
                segment,
                quantsOffset,
                highBitsOffset,
                vector,
                vectorBase,
                1,
                2,
                0,
                1,
                d * scaleK4Packed(scalesLow, scalesHigh, 0),
                d * scaleK4Packed(scalesLow, scalesHigh, 1));
        float sum1 = dotRowQ5KNoMinSuperBlock(
                segment,
                quantsOffset + 32,
                highBitsOffset,
                vector,
                vectorBase + 64,
                4,
                8,
                2,
                3,
                d * scaleK4Packed(scalesLow, scalesHigh, 2),
                d * scaleK4Packed(scalesLow, scalesHigh, 3));
        float sum2 = dotRowQ5KNoMinSuperBlock(
                segment,
                quantsOffset + 64,
                highBitsOffset,
                vector,
                vectorBase + 128,
                16,
                32,
                4,
                5,
                d * scaleK4Packed(scalesLow, scalesHigh, 4),
                d * scaleK4Packed(scalesLow, scalesHigh, 5));
        float sum3 = dotRowQ5KNoMinSuperBlock(
                segment,
                quantsOffset + 96,
                highBitsOffset,
                vector,
                vectorBase + 192,
                64,
                128,
                6,
                7,
                d * scaleK4Packed(scalesLow, scalesHigh, 6),
                d * scaleK4Packed(scalesLow, scalesHigh, 7));
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotRowQ5KNoMinSuperBlock(
            MemorySegment segment,
            long quantsOffset,
            long highBitsOffset,
            float[] vector,
            int vectorBase,
            int highMaskLow,
            int highMaskHigh,
            int highShiftLow,
            int highShiftHigh,
            float d1,
            float d2) {
        float quantDot1 = 0.0f;
        float quantDot2 = 0.0f;
        for (int i = 0; i < 32; i += Long.BYTES) {
            long packedQuants = segment.get(LE_LONG, quantsOffset + i);
            long packedHighBits = segment.get(LE_LONG, highBitsOffset + i);
            int quant0 = unsignedByte(packedQuants, 0);
            int quant1 = unsignedByte(packedQuants, 8);
            int quant2 = unsignedByte(packedQuants, 16);
            int quant3 = unsignedByte(packedQuants, 24);
            int quant4 = unsignedByte(packedQuants, 32);
            int quant5 = unsignedByte(packedQuants, 40);
            int quant6 = unsignedByte(packedQuants, 48);
            int quant7 = unsignedByte(packedQuants, 56);
            int highBits0 = unsignedByte(packedHighBits, 0);
            int highBits1 = unsignedByte(packedHighBits, 8);
            int highBits2 = unsignedByte(packedHighBits, 16);
            int highBits3 = unsignedByte(packedHighBits, 24);
            int highBits4 = unsignedByte(packedHighBits, 32);
            int highBits5 = unsignedByte(packedHighBits, 40);
            int highBits6 = unsignedByte(packedHighBits, 48);
            int highBits7 = unsignedByte(packedHighBits, 56);
            quantDot1 += ((quant0 & 0x0F) + (((highBits0 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i]
                    + ((quant1 & 0x0F) + (((highBits1 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i + 1]
                    + ((quant2 & 0x0F) + (((highBits2 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i + 2]
                    + ((quant3 & 0x0F) + (((highBits3 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i + 3]
                    + ((quant4 & 0x0F) + (((highBits4 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i + 4]
                    + ((quant5 & 0x0F) + (((highBits5 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i + 5]
                    + ((quant6 & 0x0F) + (((highBits6 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i + 6]
                    + ((quant7 & 0x0F) + (((highBits7 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vectorBase + i + 7];
            quantDot2 += ((quant0 >>> 4) + (((highBits0 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i]
                    + ((quant1 >>> 4) + (((highBits1 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i + 1]
                    + ((quant2 >>> 4) + (((highBits2 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i + 2]
                    + ((quant3 >>> 4) + (((highBits3 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i + 3]
                    + ((quant4 >>> 4) + (((highBits4 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i + 4]
                    + ((quant5 >>> 4) + (((highBits5 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i + 5]
                    + ((quant6 >>> 4) + (((highBits6 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i + 6]
                    + ((quant7 >>> 4) + (((highBits7 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vectorBase + 32 + i + 7];
        }
        return d1 * quantDot1 + d2 * quantDot2;
    }

    static float dotRowQ5KWithGroupSums(
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
        int groupIndex = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ5KGroupSumBlock(segment, blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            sum1 += dotRowQ5KGroupSumBlock(
                    segment,
                    blockOffset + Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS);
            sum2 += dotRowQ5KGroupSumBlock(
                    segment,
                    blockOffset + 2L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS);
            sum3 += dotRowQ5KGroupSumBlock(
                    segment,
                    blockOffset + 3L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS);
            blockOffset += 4L * Q5_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
            groupIndex += 4 * QK_GROUPS;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ5KGroupSumBlock(segment, blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            blockOffset += Q5_K_BLOCK_BYTES;
            vectorBase += QK_K;
            groupIndex += QK_GROUPS;
        }
        return sum;
    }

    static void dotRowsQ5KWithGroupSums4(
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
        int groupIndex = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ5KGroupSumBlock(segment, row0Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row1Sum0 += dotRowQ5KGroupSumBlock(segment, row1Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row2Sum0 += dotRowQ5KGroupSumBlock(segment, row2Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row3Sum0 += dotRowQ5KGroupSumBlock(segment, row3Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row0Sum1 += dotRowQ5KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS);
            row1Sum1 += dotRowQ5KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS);
            row2Sum1 += dotRowQ5KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS);
            row3Sum1 += dotRowQ5KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS);
            row0Sum2 += dotRowQ5KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS);
            row1Sum2 += dotRowQ5KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS);
            row2Sum2 += dotRowQ5KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS);
            row3Sum2 += dotRowQ5KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + 2L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS);
            row0Sum3 += dotRowQ5KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS);
            row1Sum3 += dotRowQ5KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS);
            row2Sum3 += dotRowQ5KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS);
            row3Sum3 += dotRowQ5KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + 3L * Q5_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS);
            blockOffset += 4L * Q5_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
            groupIndex += 4 * QK_GROUPS;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ5KGroupSumBlock(segment, row0Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row1Sum += dotRowQ5KGroupSumBlock(segment, row1Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row2Sum += dotRowQ5KGroupSumBlock(segment, row2Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row3Sum += dotRowQ5KGroupSumBlock(segment, row3Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            blockOffset += Q5_K_BLOCK_BYTES;
            vectorBase += QK_K;
            groupIndex += QK_GROUPS;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowQ5KGroupSumBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase,
            float[] vectorGroupSums,
            int groupIndex) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        boolean hasBlockMins = dMin != 0.0f;
        long scalesOffset = blockOffset + 4;
        long scalesLow = segment.get(LE_LONG, scalesOffset);
        int scalesHigh = segment.get(LE_INT, scalesOffset + Long.BYTES);
        long highBitsOffset = blockOffset + 16;
        long quantsOffset = blockOffset + 48;
        float sum0 = dotRowQ5KGroupSumSuperBlock(
                segment,
                quantsOffset,
                highBitsOffset,
                vector,
                vectorBase,
                vectorGroupSums,
                groupIndex,
                1,
                2,
                0,
                1,
                d,
                dMin,
                hasBlockMins,
                scalesLow,
                scalesHigh,
                0,
                1);
        float sum1 = dotRowQ5KGroupSumSuperBlock(
                segment,
                quantsOffset + 32,
                highBitsOffset,
                vector,
                vectorBase + 64,
                vectorGroupSums,
                groupIndex + 2,
                4,
                8,
                2,
                3,
                d,
                dMin,
                hasBlockMins,
                scalesLow,
                scalesHigh,
                2,
                3);
        float sum2 = dotRowQ5KGroupSumSuperBlock(
                segment,
                quantsOffset + 64,
                highBitsOffset,
                vector,
                vectorBase + 128,
                vectorGroupSums,
                groupIndex + 4,
                16,
                32,
                4,
                5,
                d,
                dMin,
                hasBlockMins,
                scalesLow,
                scalesHigh,
                4,
                5);
        float sum3 = dotRowQ5KGroupSumSuperBlock(
                segment,
                quantsOffset + 96,
                highBitsOffset,
                vector,
                vectorBase + 192,
                vectorGroupSums,
                groupIndex + 6,
                64,
                128,
                6,
                7,
                d,
                dMin,
                hasBlockMins,
                scalesLow,
                scalesHigh,
                6,
                7);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotRowQ5KGroupSumSuperBlock(
            MemorySegment segment,
            long quantsOffset,
            long highBitsOffset,
            float[] vector,
            int vBase,
            float[] vectorGroupSums,
            int groupIndex,
            int highMaskLow,
            int highMaskHigh,
            int highShiftLow,
            int highShiftHigh,
            float d,
            float dMin,
            boolean hasBlockMins,
            long scalesLow,
            int scalesHigh,
            int firstScaleIndex,
            int secondScaleIndex) {
        boolean hasPairMins = false;
        float d1;
        float d2;
        float m1 = 0.0f;
        float m2 = 0.0f;
        if (hasBlockMins) {
            int first = scaleMinK4PackedCode(scalesLow, scalesHigh, firstScaleIndex);
            int second = scaleMinK4PackedCode(scalesLow, scalesHigh, secondScaleIndex);
            int firstMin = minFromPackedScaleMin(first);
            int secondMin = minFromPackedScaleMin(second);
            d1 = d * scaleFromPackedScaleMin(first);
            d2 = d * scaleFromPackedScaleMin(second);
            hasPairMins = firstMin != 0 || secondMin != 0;
            if (hasPairMins) {
                m1 = minContribution(dMin, firstMin);
                m2 = minContribution(dMin, secondMin);
            }
        } else {
            d1 = d * scaleK4Packed(scalesLow, scalesHigh, firstScaleIndex);
            d2 = d * scaleK4Packed(scalesLow, scalesHigh, secondScaleIndex);
        }

        float quantDot1 = 0.0f;
        float quantDot2 = 0.0f;
        for (int i = 0; i < 32; i += Long.BYTES) {
            long packedQuants = segment.get(LE_LONG, quantsOffset + i);
            long packedHighBits = segment.get(LE_LONG, highBitsOffset + i);
            int quant0 = unsignedByte(packedQuants, 0);
            int quant1 = unsignedByte(packedQuants, 8);
            int quant2 = unsignedByte(packedQuants, 16);
            int quant3 = unsignedByte(packedQuants, 24);
            int quant4 = unsignedByte(packedQuants, 32);
            int quant5 = unsignedByte(packedQuants, 40);
            int quant6 = unsignedByte(packedQuants, 48);
            int quant7 = unsignedByte(packedQuants, 56);
            int highBits0 = unsignedByte(packedHighBits, 0);
            int highBits1 = unsignedByte(packedHighBits, 8);
            int highBits2 = unsignedByte(packedHighBits, 16);
            int highBits3 = unsignedByte(packedHighBits, 24);
            int highBits4 = unsignedByte(packedHighBits, 32);
            int highBits5 = unsignedByte(packedHighBits, 40);
            int highBits6 = unsignedByte(packedHighBits, 48);
            int highBits7 = unsignedByte(packedHighBits, 56);
            quantDot1 += ((quant0 & 0x0F) + (((highBits0 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i]
                    + ((quant1 & 0x0F) + (((highBits1 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i + 1]
                    + ((quant2 & 0x0F) + (((highBits2 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i + 2]
                    + ((quant3 & 0x0F) + (((highBits3 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i + 3]
                    + ((quant4 & 0x0F) + (((highBits4 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i + 4]
                    + ((quant5 & 0x0F) + (((highBits5 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i + 5]
                    + ((quant6 & 0x0F) + (((highBits6 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i + 6]
                    + ((quant7 & 0x0F) + (((highBits7 & highMaskLow) >>> highShiftLow) << 4))
                            * vector[vBase + i + 7];
            quantDot2 += ((quant0 >>> 4) + (((highBits0 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i]
                    + ((quant1 >>> 4) + (((highBits1 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i + 1]
                    + ((quant2 >>> 4) + (((highBits2 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i + 2]
                    + ((quant3 >>> 4) + (((highBits3 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i + 3]
                    + ((quant4 >>> 4) + (((highBits4 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i + 4]
                    + ((quant5 >>> 4) + (((highBits5 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i + 5]
                    + ((quant6 >>> 4) + (((highBits6 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i + 6]
                    + ((quant7 >>> 4) + (((highBits7 & highMaskHigh) >>> highShiftHigh) << 4))
                            * vector[vBase + 32 + i + 7];
        }
        float contribution = d1 * quantDot1 + d2 * quantDot2;
        if (hasPairMins) {
            contribution -= m1 * vectorGroupSums[groupIndex] + m2 * vectorGroupSums[groupIndex + 1];
        }
        return contribution;
    }
}
