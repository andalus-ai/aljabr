package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q3_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.q3KHighBias;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.unpackQ3KScales;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw Q3_K row-dot kernel.
 *
 * <p>Q3_K stores two-bit quants with separate high-bit masks and signed group
 * scales. This helper keeps that decoding isolated from the public tensor
 * facade and raw row scheduling.</p>
 * @author bhangun
 */
final class GgufQ3RawDot {
    private static final ThreadLocal<int[]> SCALES = ThreadLocal.withInitial(() -> new int[16]);
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ3RawDot() {
    }

    static float dotRowQ3K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        return dotRowQ3K(segment, rowOffset, columns, vector, vectorOffset, SCALES.get());
    }

    static float dotRowQ3K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset,
            int[] scales) {
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
            sum0 += dotRowQ3KBlock(segment, blockOffset, vector, vectorBase, scales);
            sum1 += dotRowQ3KBlock(
                    segment,
                    blockOffset + Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    scales);
            sum2 += dotRowQ3KBlock(
                    segment,
                    blockOffset + 2L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    scales);
            sum3 += dotRowQ3KBlock(
                    segment,
                    blockOffset + 3L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    scales);
            blockOffset += 4L * Q3_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ3KBlock(segment, blockOffset, vector, vectorBase, scales);
            blockOffset += Q3_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return sum;
    }

    static void dotRowsQ3K4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset,
            int[] scales) {
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
            row0Sum0 += dotRowQ3KBlock(segment, row0Offset + blockOffset, vector, vectorBase, scales);
            row1Sum0 += dotRowQ3KBlock(segment, row1Offset + blockOffset, vector, vectorBase, scales);
            row2Sum0 += dotRowQ3KBlock(segment, row2Offset + blockOffset, vector, vectorBase, scales);
            row3Sum0 += dotRowQ3KBlock(segment, row3Offset + blockOffset, vector, vectorBase, scales);
            row0Sum1 += dotRowQ3KBlock(
                    segment,
                    row0Offset + blockOffset + Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    scales);
            row1Sum1 += dotRowQ3KBlock(
                    segment,
                    row1Offset + blockOffset + Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    scales);
            row2Sum1 += dotRowQ3KBlock(
                    segment,
                    row2Offset + blockOffset + Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    scales);
            row3Sum1 += dotRowQ3KBlock(
                    segment,
                    row3Offset + blockOffset + Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    scales);
            row0Sum2 += dotRowQ3KBlock(
                    segment,
                    row0Offset + blockOffset + 2L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    scales);
            row1Sum2 += dotRowQ3KBlock(
                    segment,
                    row1Offset + blockOffset + 2L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    scales);
            row2Sum2 += dotRowQ3KBlock(
                    segment,
                    row2Offset + blockOffset + 2L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    scales);
            row3Sum2 += dotRowQ3KBlock(
                    segment,
                    row3Offset + blockOffset + 2L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    scales);
            row0Sum3 += dotRowQ3KBlock(
                    segment,
                    row0Offset + blockOffset + 3L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    scales);
            row1Sum3 += dotRowQ3KBlock(
                    segment,
                    row1Offset + blockOffset + 3L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    scales);
            row2Sum3 += dotRowQ3KBlock(
                    segment,
                    row2Offset + blockOffset + 3L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    scales);
            row3Sum3 += dotRowQ3KBlock(
                    segment,
                    row3Offset + blockOffset + 3L * Q3_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    scales);
            blockOffset += 4L * Q3_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ3KBlock(segment, row0Offset + blockOffset, vector, vectorBase, scales);
            row1Sum += dotRowQ3KBlock(segment, row1Offset + blockOffset, vector, vectorBase, scales);
            row2Sum += dotRowQ3KBlock(segment, row2Offset + blockOffset, vector, vectorBase, scales);
            row3Sum += dotRowQ3KBlock(segment, row3Offset + blockOffset, vector, vectorBase, scales);
            blockOffset += Q3_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowQ3KBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase,
            int[] scales) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 108));
        long hmaskOffset = blockOffset;
        long packedBase = blockOffset + 32;
        unpackQ3KScales(segment, blockOffset + 96, scales);
        return dotRowQ3KSuperBlock(
                segment,
                hmaskOffset,
                packedBase,
                vector,
                vectorBase,
                d,
                scales,
                0,
                1)
                + dotRowQ3KSuperBlock(
                        segment,
                        hmaskOffset,
                        packedBase + 32,
                        vector,
                        vectorBase + 128,
                        d,
                        scales,
                        8,
                        16);
    }

    private static float dotRowQ3KSuperBlock(
            MemorySegment segment,
            long hmaskOffset,
            long packedBase,
            float[] vector,
            int vectorBase,
            float d,
            int[] scales,
            int scaleIndex,
            int highMask) {
        int mask0 = highMask;
        int mask1 = mask0 << 1;
        int mask2 = mask1 << 1;
        int mask3 = mask2 << 1;
        int scale0 = scales[scaleIndex];
        int scale1 = scales[scaleIndex + 1];
        int scale2 = scales[scaleIndex + 2];
        int scale3 = scales[scaleIndex + 3];
        int scale4 = scales[scaleIndex + 4];
        int scale5 = scales[scaleIndex + 5];
        int scale6 = scales[scaleIndex + 6];
        int scale7 = scales[scaleIndex + 7];
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        float dot4 = 0.0f;
        float dot5 = 0.0f;
        float dot6 = 0.0f;
        float dot7 = 0.0f;
        for (int i = 0; i < 16; i += Long.BYTES) {
            long firstQuants = segment.get(LE_LONG, packedBase + i);
            long secondQuants = segment.get(LE_LONG, packedBase + 16 + i);
            long firstHighs = segment.get(LE_LONG, hmaskOffset + i);
            long secondHighs = segment.get(LE_LONG, hmaskOffset + 16 + i);
            int firstQuant0 = unsignedByte(firstQuants, 0);
            int firstQuant1 = unsignedByte(firstQuants, 8);
            int firstQuant2 = unsignedByte(firstQuants, 16);
            int firstQuant3 = unsignedByte(firstQuants, 24);
            int firstQuant4 = unsignedByte(firstQuants, 32);
            int firstQuant5 = unsignedByte(firstQuants, 40);
            int firstQuant6 = unsignedByte(firstQuants, 48);
            int firstQuant7 = unsignedByte(firstQuants, 56);
            int secondQuant0 = unsignedByte(secondQuants, 0);
            int secondQuant1 = unsignedByte(secondQuants, 8);
            int secondQuant2 = unsignedByte(secondQuants, 16);
            int secondQuant3 = unsignedByte(secondQuants, 24);
            int secondQuant4 = unsignedByte(secondQuants, 32);
            int secondQuant5 = unsignedByte(secondQuants, 40);
            int secondQuant6 = unsignedByte(secondQuants, 48);
            int secondQuant7 = unsignedByte(secondQuants, 56);
            int firstHigh0 = unsignedByte(firstHighs, 0);
            int firstHigh1 = unsignedByte(firstHighs, 8);
            int firstHigh2 = unsignedByte(firstHighs, 16);
            int firstHigh3 = unsignedByte(firstHighs, 24);
            int firstHigh4 = unsignedByte(firstHighs, 32);
            int firstHigh5 = unsignedByte(firstHighs, 40);
            int firstHigh6 = unsignedByte(firstHighs, 48);
            int firstHigh7 = unsignedByte(firstHighs, 56);
            int secondHigh0 = unsignedByte(secondHighs, 0);
            int secondHigh1 = unsignedByte(secondHighs, 8);
            int secondHigh2 = unsignedByte(secondHighs, 16);
            int secondHigh3 = unsignedByte(secondHighs, 24);
            int secondHigh4 = unsignedByte(secondHighs, 32);
            int secondHigh5 = unsignedByte(secondHighs, 40);
            int secondHigh6 = unsignedByte(secondHighs, 48);
            int secondHigh7 = unsignedByte(secondHighs, 56);
            dot0 += ((firstQuant0 & 0x03) - q3KHighBias(firstHigh0, mask0)) * vector[vectorBase + i]
                    + ((firstQuant1 & 0x03) - q3KHighBias(firstHigh1, mask0))
                            * vector[vectorBase + i + 1]
                    + ((firstQuant2 & 0x03) - q3KHighBias(firstHigh2, mask0))
                            * vector[vectorBase + i + 2]
                    + ((firstQuant3 & 0x03) - q3KHighBias(firstHigh3, mask0))
                            * vector[vectorBase + i + 3]
                    + ((firstQuant4 & 0x03) - q3KHighBias(firstHigh4, mask0))
                            * vector[vectorBase + i + 4]
                    + ((firstQuant5 & 0x03) - q3KHighBias(firstHigh5, mask0))
                            * vector[vectorBase + i + 5]
                    + ((firstQuant6 & 0x03) - q3KHighBias(firstHigh6, mask0))
                            * vector[vectorBase + i + 6]
                    + ((firstQuant7 & 0x03) - q3KHighBias(firstHigh7, mask0))
                            * vector[vectorBase + i + 7];
            dot1 += ((secondQuant0 & 0x03) - q3KHighBias(secondHigh0, mask0))
                            * vector[vectorBase + 16 + i]
                    + ((secondQuant1 & 0x03) - q3KHighBias(secondHigh1, mask0))
                            * vector[vectorBase + 16 + i + 1]
                    + ((secondQuant2 & 0x03) - q3KHighBias(secondHigh2, mask0))
                            * vector[vectorBase + 16 + i + 2]
                    + ((secondQuant3 & 0x03) - q3KHighBias(secondHigh3, mask0))
                            * vector[vectorBase + 16 + i + 3]
                    + ((secondQuant4 & 0x03) - q3KHighBias(secondHigh4, mask0))
                            * vector[vectorBase + 16 + i + 4]
                    + ((secondQuant5 & 0x03) - q3KHighBias(secondHigh5, mask0))
                            * vector[vectorBase + 16 + i + 5]
                    + ((secondQuant6 & 0x03) - q3KHighBias(secondHigh6, mask0))
                            * vector[vectorBase + 16 + i + 6]
                    + ((secondQuant7 & 0x03) - q3KHighBias(secondHigh7, mask0))
                            * vector[vectorBase + 16 + i + 7];
            dot2 += (((firstQuant0 >>> 2) & 0x03) - q3KHighBias(firstHigh0, mask1))
                            * vector[vectorBase + 32 + i]
                    + (((firstQuant1 >>> 2) & 0x03) - q3KHighBias(firstHigh1, mask1))
                            * vector[vectorBase + 32 + i + 1]
                    + (((firstQuant2 >>> 2) & 0x03) - q3KHighBias(firstHigh2, mask1))
                            * vector[vectorBase + 32 + i + 2]
                    + (((firstQuant3 >>> 2) & 0x03) - q3KHighBias(firstHigh3, mask1))
                            * vector[vectorBase + 32 + i + 3]
                    + (((firstQuant4 >>> 2) & 0x03) - q3KHighBias(firstHigh4, mask1))
                            * vector[vectorBase + 32 + i + 4]
                    + (((firstQuant5 >>> 2) & 0x03) - q3KHighBias(firstHigh5, mask1))
                            * vector[vectorBase + 32 + i + 5]
                    + (((firstQuant6 >>> 2) & 0x03) - q3KHighBias(firstHigh6, mask1))
                            * vector[vectorBase + 32 + i + 6]
                    + (((firstQuant7 >>> 2) & 0x03) - q3KHighBias(firstHigh7, mask1))
                            * vector[vectorBase + 32 + i + 7];
            dot3 += (((secondQuant0 >>> 2) & 0x03) - q3KHighBias(secondHigh0, mask1))
                            * vector[vectorBase + 48 + i]
                    + (((secondQuant1 >>> 2) & 0x03) - q3KHighBias(secondHigh1, mask1))
                            * vector[vectorBase + 48 + i + 1]
                    + (((secondQuant2 >>> 2) & 0x03) - q3KHighBias(secondHigh2, mask1))
                            * vector[vectorBase + 48 + i + 2]
                    + (((secondQuant3 >>> 2) & 0x03) - q3KHighBias(secondHigh3, mask1))
                            * vector[vectorBase + 48 + i + 3]
                    + (((secondQuant4 >>> 2) & 0x03) - q3KHighBias(secondHigh4, mask1))
                            * vector[vectorBase + 48 + i + 4]
                    + (((secondQuant5 >>> 2) & 0x03) - q3KHighBias(secondHigh5, mask1))
                            * vector[vectorBase + 48 + i + 5]
                    + (((secondQuant6 >>> 2) & 0x03) - q3KHighBias(secondHigh6, mask1))
                            * vector[vectorBase + 48 + i + 6]
                    + (((secondQuant7 >>> 2) & 0x03) - q3KHighBias(secondHigh7, mask1))
                            * vector[vectorBase + 48 + i + 7];
            dot4 += (((firstQuant0 >>> 4) & 0x03) - q3KHighBias(firstHigh0, mask2))
                            * vector[vectorBase + 64 + i]
                    + (((firstQuant1 >>> 4) & 0x03) - q3KHighBias(firstHigh1, mask2))
                            * vector[vectorBase + 64 + i + 1]
                    + (((firstQuant2 >>> 4) & 0x03) - q3KHighBias(firstHigh2, mask2))
                            * vector[vectorBase + 64 + i + 2]
                    + (((firstQuant3 >>> 4) & 0x03) - q3KHighBias(firstHigh3, mask2))
                            * vector[vectorBase + 64 + i + 3]
                    + (((firstQuant4 >>> 4) & 0x03) - q3KHighBias(firstHigh4, mask2))
                            * vector[vectorBase + 64 + i + 4]
                    + (((firstQuant5 >>> 4) & 0x03) - q3KHighBias(firstHigh5, mask2))
                            * vector[vectorBase + 64 + i + 5]
                    + (((firstQuant6 >>> 4) & 0x03) - q3KHighBias(firstHigh6, mask2))
                            * vector[vectorBase + 64 + i + 6]
                    + (((firstQuant7 >>> 4) & 0x03) - q3KHighBias(firstHigh7, mask2))
                            * vector[vectorBase + 64 + i + 7];
            dot5 += (((secondQuant0 >>> 4) & 0x03) - q3KHighBias(secondHigh0, mask2))
                            * vector[vectorBase + 80 + i]
                    + (((secondQuant1 >>> 4) & 0x03) - q3KHighBias(secondHigh1, mask2))
                            * vector[vectorBase + 80 + i + 1]
                    + (((secondQuant2 >>> 4) & 0x03) - q3KHighBias(secondHigh2, mask2))
                            * vector[vectorBase + 80 + i + 2]
                    + (((secondQuant3 >>> 4) & 0x03) - q3KHighBias(secondHigh3, mask2))
                            * vector[vectorBase + 80 + i + 3]
                    + (((secondQuant4 >>> 4) & 0x03) - q3KHighBias(secondHigh4, mask2))
                            * vector[vectorBase + 80 + i + 4]
                    + (((secondQuant5 >>> 4) & 0x03) - q3KHighBias(secondHigh5, mask2))
                            * vector[vectorBase + 80 + i + 5]
                    + (((secondQuant6 >>> 4) & 0x03) - q3KHighBias(secondHigh6, mask2))
                            * vector[vectorBase + 80 + i + 6]
                    + (((secondQuant7 >>> 4) & 0x03) - q3KHighBias(secondHigh7, mask2))
                            * vector[vectorBase + 80 + i + 7];
            dot6 += (((firstQuant0 >>> 6) & 0x03) - q3KHighBias(firstHigh0, mask3))
                            * vector[vectorBase + 96 + i]
                    + (((firstQuant1 >>> 6) & 0x03) - q3KHighBias(firstHigh1, mask3))
                            * vector[vectorBase + 96 + i + 1]
                    + (((firstQuant2 >>> 6) & 0x03) - q3KHighBias(firstHigh2, mask3))
                            * vector[vectorBase + 96 + i + 2]
                    + (((firstQuant3 >>> 6) & 0x03) - q3KHighBias(firstHigh3, mask3))
                            * vector[vectorBase + 96 + i + 3]
                    + (((firstQuant4 >>> 6) & 0x03) - q3KHighBias(firstHigh4, mask3))
                            * vector[vectorBase + 96 + i + 4]
                    + (((firstQuant5 >>> 6) & 0x03) - q3KHighBias(firstHigh5, mask3))
                            * vector[vectorBase + 96 + i + 5]
                    + (((firstQuant6 >>> 6) & 0x03) - q3KHighBias(firstHigh6, mask3))
                            * vector[vectorBase + 96 + i + 6]
                    + (((firstQuant7 >>> 6) & 0x03) - q3KHighBias(firstHigh7, mask3))
                            * vector[vectorBase + 96 + i + 7];
            dot7 += (((secondQuant0 >>> 6) & 0x03) - q3KHighBias(secondHigh0, mask3))
                            * vector[vectorBase + 112 + i]
                    + (((secondQuant1 >>> 6) & 0x03) - q3KHighBias(secondHigh1, mask3))
                            * vector[vectorBase + 112 + i + 1]
                    + (((secondQuant2 >>> 6) & 0x03) - q3KHighBias(secondHigh2, mask3))
                            * vector[vectorBase + 112 + i + 2]
                    + (((secondQuant3 >>> 6) & 0x03) - q3KHighBias(secondHigh3, mask3))
                            * vector[vectorBase + 112 + i + 3]
                    + (((secondQuant4 >>> 6) & 0x03) - q3KHighBias(secondHigh4, mask3))
                            * vector[vectorBase + 112 + i + 4]
                    + (((secondQuant5 >>> 6) & 0x03) - q3KHighBias(secondHigh5, mask3))
                            * vector[vectorBase + 112 + i + 5]
                    + (((secondQuant6 >>> 6) & 0x03) - q3KHighBias(secondHigh6, mask3))
                            * vector[vectorBase + 112 + i + 6]
                    + (((secondQuant7 >>> 6) & 0x03) - q3KHighBias(secondHigh7, mask3))
                            * vector[vectorBase + 112 + i + 7];
        }
        return d * (scale0 * dot0 + scale1 * dot1 + scale2 * dot2 + scale3 * dot3
                + scale4 * dot4 + scale5 * dot5 + scale6 * dot6 + scale7 * dot7);
    }
}
