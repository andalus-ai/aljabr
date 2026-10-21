package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Block-level dequantizers for GGUF K-family quantization formats.
 *
 * <p>The K formats share super-block scale/min packing but differ in low-bit
 * and high-bit layouts. Keeping those kernels here leaves {@link GgufTensorOps}
 * focused on orchestration and row/matrix traversal.</p>
 * @author bhangun
 */
final class GgufKBlockDequantizer {
    private static final ThreadLocal<int[]> Q3K_SCALES = ThreadLocal.withInitial(() -> new int[16]);
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufKBlockDequantizer() {
    }

    static void dequantizeQ4KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        long scalesOffset = blockOffset + 4;
        long scalesLow = segment.get(LE_LONG, scalesOffset);
        int scalesHigh = segment.get(LE_INT, scalesOffset + Long.BYTES);
        long quantsOffset = blockOffset + 16;
        int outBase = dstOffset;
        int scaleIndex = 0;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            int firstScaleIndex = scaleIndex++;
            int secondScaleIndex = scaleIndex++;
            float d1;
            float d2;
            float m1 = 0.0f;
            float m2 = 0.0f;
            if (dMin == 0.0f) {
                d1 = d * scaleK4Packed(scalesLow, scalesHigh, firstScaleIndex);
                d2 = d * scaleK4Packed(scalesLow, scalesHigh, secondScaleIndex);
            } else {
                int first = scaleMinK4PackedCode(scalesLow, scalesHigh, firstScaleIndex);
                int second = scaleMinK4PackedCode(scalesLow, scalesHigh, secondScaleIndex);
                d1 = d * scaleFromPackedScaleMin(first);
                d2 = d * scaleFromPackedScaleMin(second);
                m1 = minContribution(dMin, minFromPackedScaleMin(first));
                m2 = minContribution(dMin, minFromPackedScaleMin(second));
            }

            for (int i = 0; i < 32; i += Long.BYTES) {
                long packed = segment.get(LE_LONG, quantsOffset + i);
                int quant0 = unsignedByte(packed, 0);
                int quant1 = unsignedByte(packed, 8);
                int quant2 = unsignedByte(packed, 16);
                int quant3 = unsignedByte(packed, 24);
                int quant4 = unsignedByte(packed, 32);
                int quant5 = unsignedByte(packed, 40);
                int quant6 = unsignedByte(packed, 48);
                int quant7 = unsignedByte(packed, 56);
                dst[outBase + superBlockOffset + i] = d1 * (quant0 & 0x0F) - m1;
                dst[outBase + superBlockOffset + i + 1] = d1 * (quant1 & 0x0F) - m1;
                dst[outBase + superBlockOffset + i + 2] = d1 * (quant2 & 0x0F) - m1;
                dst[outBase + superBlockOffset + i + 3] = d1 * (quant3 & 0x0F) - m1;
                dst[outBase + superBlockOffset + i + 4] = d1 * (quant4 & 0x0F) - m1;
                dst[outBase + superBlockOffset + i + 5] = d1 * (quant5 & 0x0F) - m1;
                dst[outBase + superBlockOffset + i + 6] = d1 * (quant6 & 0x0F) - m1;
                dst[outBase + superBlockOffset + i + 7] = d1 * (quant7 & 0x0F) - m1;
                dst[outBase + superBlockOffset + 32 + i] = d2 * (quant0 >>> 4) - m2;
                dst[outBase + superBlockOffset + 32 + i + 1] = d2 * (quant1 >>> 4) - m2;
                dst[outBase + superBlockOffset + 32 + i + 2] = d2 * (quant2 >>> 4) - m2;
                dst[outBase + superBlockOffset + 32 + i + 3] = d2 * (quant3 >>> 4) - m2;
                dst[outBase + superBlockOffset + 32 + i + 4] = d2 * (quant4 >>> 4) - m2;
                dst[outBase + superBlockOffset + 32 + i + 5] = d2 * (quant5 >>> 4) - m2;
                dst[outBase + superBlockOffset + 32 + i + 6] = d2 * (quant6 >>> 4) - m2;
                dst[outBase + superBlockOffset + 32 + i + 7] = d2 * (quant7 >>> 4) - m2;
            }
            quantsOffset += 32;
        }
    }

    static void dequantizeQ5KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        long scalesOffset = blockOffset + 4;
        long scalesLow = segment.get(LE_LONG, scalesOffset);
        int scalesHigh = segment.get(LE_INT, scalesOffset + Long.BYTES);
        long highBitsOffset = blockOffset + 16;
        long quantsOffset = blockOffset + 48;
        int scaleIndex = 0;
        int highMaskLow = 1;
        int highMaskHigh = 2;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            int firstScaleIndex = scaleIndex++;
            int secondScaleIndex = scaleIndex++;
            float d1;
            float d2;
            float m1 = 0.0f;
            float m2 = 0.0f;
            if (dMin == 0.0f) {
                d1 = d * scaleK4Packed(scalesLow, scalesHigh, firstScaleIndex);
                d2 = d * scaleK4Packed(scalesLow, scalesHigh, secondScaleIndex);
            } else {
                int first = scaleMinK4PackedCode(scalesLow, scalesHigh, firstScaleIndex);
                int second = scaleMinK4PackedCode(scalesLow, scalesHigh, secondScaleIndex);
                d1 = d * scaleFromPackedScaleMin(first);
                d2 = d * scaleFromPackedScaleMin(second);
                m1 = minContribution(dMin, minFromPackedScaleMin(first));
                m2 = minContribution(dMin, minFromPackedScaleMin(second));
            }

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
                int low0 = (quant0 & 0x0F) + ((highBits0 & highMaskLow) != 0 ? 16 : 0);
                int low1 = (quant1 & 0x0F) + ((highBits1 & highMaskLow) != 0 ? 16 : 0);
                int low2 = (quant2 & 0x0F) + ((highBits2 & highMaskLow) != 0 ? 16 : 0);
                int low3 = (quant3 & 0x0F) + ((highBits3 & highMaskLow) != 0 ? 16 : 0);
                int low4 = (quant4 & 0x0F) + ((highBits4 & highMaskLow) != 0 ? 16 : 0);
                int low5 = (quant5 & 0x0F) + ((highBits5 & highMaskLow) != 0 ? 16 : 0);
                int low6 = (quant6 & 0x0F) + ((highBits6 & highMaskLow) != 0 ? 16 : 0);
                int low7 = (quant7 & 0x0F) + ((highBits7 & highMaskLow) != 0 ? 16 : 0);
                int high0 = (quant0 >>> 4) + ((highBits0 & highMaskHigh) != 0 ? 16 : 0);
                int high1 = (quant1 >>> 4) + ((highBits1 & highMaskHigh) != 0 ? 16 : 0);
                int high2 = (quant2 >>> 4) + ((highBits2 & highMaskHigh) != 0 ? 16 : 0);
                int high3 = (quant3 >>> 4) + ((highBits3 & highMaskHigh) != 0 ? 16 : 0);
                int high4 = (quant4 >>> 4) + ((highBits4 & highMaskHigh) != 0 ? 16 : 0);
                int high5 = (quant5 >>> 4) + ((highBits5 & highMaskHigh) != 0 ? 16 : 0);
                int high6 = (quant6 >>> 4) + ((highBits6 & highMaskHigh) != 0 ? 16 : 0);
                int high7 = (quant7 >>> 4) + ((highBits7 & highMaskHigh) != 0 ? 16 : 0);
                dst[dstOffset + superBlockOffset + i] = d1 * low0 - m1;
                dst[dstOffset + superBlockOffset + i + 1] = d1 * low1 - m1;
                dst[dstOffset + superBlockOffset + i + 2] = d1 * low2 - m1;
                dst[dstOffset + superBlockOffset + i + 3] = d1 * low3 - m1;
                dst[dstOffset + superBlockOffset + i + 4] = d1 * low4 - m1;
                dst[dstOffset + superBlockOffset + i + 5] = d1 * low5 - m1;
                dst[dstOffset + superBlockOffset + i + 6] = d1 * low6 - m1;
                dst[dstOffset + superBlockOffset + i + 7] = d1 * low7 - m1;
                dst[dstOffset + superBlockOffset + 32 + i] = d2 * high0 - m2;
                dst[dstOffset + superBlockOffset + 32 + i + 1] = d2 * high1 - m2;
                dst[dstOffset + superBlockOffset + 32 + i + 2] = d2 * high2 - m2;
                dst[dstOffset + superBlockOffset + 32 + i + 3] = d2 * high3 - m2;
                dst[dstOffset + superBlockOffset + 32 + i + 4] = d2 * high4 - m2;
                dst[dstOffset + superBlockOffset + 32 + i + 5] = d2 * high5 - m2;
                dst[dstOffset + superBlockOffset + 32 + i + 6] = d2 * high6 - m2;
                dst[dstOffset + superBlockOffset + 32 + i + 7] = d2 * high7 - m2;
            }
            quantsOffset += 32;
            highMaskLow <<= 2;
            highMaskHigh <<= 2;
        }
    }

    static void dequantizeQ2KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 80));
        float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 82));
        long scalesOffset = blockOffset;
        long quantsOffset = blockOffset + 16;
        int scaleIndex = 0;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
            long packedBase = quantsOffset + superBlockOffset / 4L;
            long packedScales = segment.get(LE_LONG, scalesOffset + scaleIndex);
            int shift = 0;
            for (int pair = 0; pair < 4; pair++) {
                int scaleShift = pair * 16;
                int firstScale = unsignedByte(packedScales, scaleShift);
                float firstD = d * (firstScale & 0x0F);
                float firstMin = minContribution(dMin, firstScale >>> 4);
                int secondScale = unsignedByte(packedScales, scaleShift + 8);
                float secondD = d * (secondScale & 0x0F);
                float secondMin = minContribution(dMin, secondScale >>> 4);
                int outBase = dstOffset + superBlockOffset + pair * 32;

                for (int i = 0; i < 16; i += Long.BYTES) {
                    long firstQuants = segment.get(LE_LONG, packedBase + i);
                    long secondQuants = segment.get(LE_LONG, packedBase + 16 + i);
                    dst[outBase + i] = firstD * ((unsignedByte(firstQuants, 0) >>> shift) & 0x03) - firstMin;
                    dst[outBase + i + 1] =
                            firstD * ((unsignedByte(firstQuants, 8) >>> shift) & 0x03) - firstMin;
                    dst[outBase + i + 2] =
                            firstD * ((unsignedByte(firstQuants, 16) >>> shift) & 0x03) - firstMin;
                    dst[outBase + i + 3] =
                            firstD * ((unsignedByte(firstQuants, 24) >>> shift) & 0x03) - firstMin;
                    dst[outBase + i + 4] =
                            firstD * ((unsignedByte(firstQuants, 32) >>> shift) & 0x03) - firstMin;
                    dst[outBase + i + 5] =
                            firstD * ((unsignedByte(firstQuants, 40) >>> shift) & 0x03) - firstMin;
                    dst[outBase + i + 6] =
                            firstD * ((unsignedByte(firstQuants, 48) >>> shift) & 0x03) - firstMin;
                    dst[outBase + i + 7] =
                            firstD * ((unsignedByte(firstQuants, 56) >>> shift) & 0x03) - firstMin;
                    dst[outBase + 16 + i] =
                            secondD * ((unsignedByte(secondQuants, 0) >>> shift) & 0x03) - secondMin;
                    dst[outBase + 16 + i + 1] =
                            secondD * ((unsignedByte(secondQuants, 8) >>> shift) & 0x03) - secondMin;
                    dst[outBase + 16 + i + 2] =
                            secondD * ((unsignedByte(secondQuants, 16) >>> shift) & 0x03) - secondMin;
                    dst[outBase + 16 + i + 3] =
                            secondD * ((unsignedByte(secondQuants, 24) >>> shift) & 0x03) - secondMin;
                    dst[outBase + 16 + i + 4] =
                            secondD * ((unsignedByte(secondQuants, 32) >>> shift) & 0x03) - secondMin;
                    dst[outBase + 16 + i + 5] =
                            secondD * ((unsignedByte(secondQuants, 40) >>> shift) & 0x03) - secondMin;
                    dst[outBase + 16 + i + 6] =
                            secondD * ((unsignedByte(secondQuants, 48) >>> shift) & 0x03) - secondMin;
                    dst[outBase + 16 + i + 7] =
                            secondD * ((unsignedByte(secondQuants, 56) >>> shift) & 0x03) - secondMin;
                }
                shift += 2;
            }
            scaleIndex += 8;
        }
    }

    static void dequantizeQ3KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 108));
        long hmaskOffset = blockOffset;
        long quantsOffset = blockOffset + 32;
        int[] scales = Q3K_SCALES.get();
        unpackQ3KScales(segment, blockOffset + 96, scales);
        int scaleIndex = 0;
        int highMask = 1;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
            long packedBase = quantsOffset + superBlockOffset / 4L;
            int shift = 0;
            for (int pair = 0; pair < 4; pair++) {
                float firstD = d * scales[scaleIndex++];
                float secondD = d * scales[scaleIndex++];
                int outBase = dstOffset + superBlockOffset + pair * 32;

                for (int i = 0; i < 16; i += Long.BYTES) {
                    long firstQuants = segment.get(LE_LONG, packedBase + i);
                    long secondQuants = segment.get(LE_LONG, packedBase + 16 + i);
                    long firstHighs = segment.get(LE_LONG, hmaskOffset + i);
                    long secondHighs = segment.get(LE_LONG, hmaskOffset + 16 + i);
                    int firstQuant0 = (unsignedByte(firstQuants, 0) >>> shift) & 0x03;
                    int firstQuant1 = (unsignedByte(firstQuants, 8) >>> shift) & 0x03;
                    int firstQuant2 = (unsignedByte(firstQuants, 16) >>> shift) & 0x03;
                    int firstQuant3 = (unsignedByte(firstQuants, 24) >>> shift) & 0x03;
                    int firstQuant4 = (unsignedByte(firstQuants, 32) >>> shift) & 0x03;
                    int firstQuant5 = (unsignedByte(firstQuants, 40) >>> shift) & 0x03;
                    int firstQuant6 = (unsignedByte(firstQuants, 48) >>> shift) & 0x03;
                    int firstQuant7 = (unsignedByte(firstQuants, 56) >>> shift) & 0x03;
                    int secondQuant0 = (unsignedByte(secondQuants, 0) >>> shift) & 0x03;
                    int secondQuant1 = (unsignedByte(secondQuants, 8) >>> shift) & 0x03;
                    int secondQuant2 = (unsignedByte(secondQuants, 16) >>> shift) & 0x03;
                    int secondQuant3 = (unsignedByte(secondQuants, 24) >>> shift) & 0x03;
                    int secondQuant4 = (unsignedByte(secondQuants, 32) >>> shift) & 0x03;
                    int secondQuant5 = (unsignedByte(secondQuants, 40) >>> shift) & 0x03;
                    int secondQuant6 = (unsignedByte(secondQuants, 48) >>> shift) & 0x03;
                    int secondQuant7 = (unsignedByte(secondQuants, 56) >>> shift) & 0x03;
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
                    dst[outBase + i] = firstD * (firstQuant0 - q3KHighBias(firstHigh0, highMask));
                    dst[outBase + i + 1] = firstD * (firstQuant1 - q3KHighBias(firstHigh1, highMask));
                    dst[outBase + i + 2] = firstD * (firstQuant2 - q3KHighBias(firstHigh2, highMask));
                    dst[outBase + i + 3] = firstD * (firstQuant3 - q3KHighBias(firstHigh3, highMask));
                    dst[outBase + i + 4] = firstD * (firstQuant4 - q3KHighBias(firstHigh4, highMask));
                    dst[outBase + i + 5] = firstD * (firstQuant5 - q3KHighBias(firstHigh5, highMask));
                    dst[outBase + i + 6] = firstD * (firstQuant6 - q3KHighBias(firstHigh6, highMask));
                    dst[outBase + i + 7] = firstD * (firstQuant7 - q3KHighBias(firstHigh7, highMask));
                    dst[outBase + 16 + i] = secondD * (secondQuant0 - q3KHighBias(secondHigh0, highMask));
                    dst[outBase + 16 + i + 1] = secondD * (secondQuant1 - q3KHighBias(secondHigh1, highMask));
                    dst[outBase + 16 + i + 2] = secondD * (secondQuant2 - q3KHighBias(secondHigh2, highMask));
                    dst[outBase + 16 + i + 3] = secondD * (secondQuant3 - q3KHighBias(secondHigh3, highMask));
                    dst[outBase + 16 + i + 4] = secondD * (secondQuant4 - q3KHighBias(secondHigh4, highMask));
                    dst[outBase + 16 + i + 5] = secondD * (secondQuant5 - q3KHighBias(secondHigh5, highMask));
                    dst[outBase + 16 + i + 6] = secondD * (secondQuant6 - q3KHighBias(secondHigh6, highMask));
                    dst[outBase + 16 + i + 7] = secondD * (secondQuant7 - q3KHighBias(secondHigh7, highMask));
                }
                shift += 2;
                highMask <<= 1;
            }
        }
    }

    static void dequantizeQ6KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 208));
        long lowBitsBase = blockOffset;
        long highBitsBase = blockOffset + 128;
        long scalesOffset = blockOffset + 192;
        int outBase = dstOffset;
        int scaleBase = 0;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
            long packedScales = segment.get(LE_LONG, scalesOffset + scaleBase);
            for (int half = 0; half < 2; half++) {
                int scaleShift = half * 8;
                float scale1 = d * signedByte(packedScales, scaleShift);
                float scale2 = d * signedByte(packedScales, scaleShift + 16);
                float scale3 = d * signedByte(packedScales, scaleShift + 32);
                float scale4 = d * signedByte(packedScales, scaleShift + 48);
                int halfEnd = half * 16 + 16;
                for (int i = half * 16; i < halfEnd; i += Long.BYTES) {
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
                    dst[outBase + i] = scale1 * (((lowA0 & 0x0F) | (((highBits0 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + i + 1] = scale1 * (((lowA1 & 0x0F) | (((highBits1 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + i + 2] = scale1 * (((lowA2 & 0x0F) | (((highBits2 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + i + 3] = scale1 * (((lowA3 & 0x0F) | (((highBits3 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + i + 4] = scale1 * (((lowA4 & 0x0F) | (((highBits4 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + i + 5] = scale1 * (((lowA5 & 0x0F) | (((highBits5 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + i + 6] = scale1 * (((lowA6 & 0x0F) | (((highBits6 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + i + 7] = scale1 * (((lowA7 & 0x0F) | (((highBits7 >>> 0) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i] =
                            scale2 * (((lowB0 & 0x0F) | (((highBits0 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i + 1] =
                            scale2 * (((lowB1 & 0x0F) | (((highBits1 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i + 2] =
                            scale2 * (((lowB2 & 0x0F) | (((highBits2 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i + 3] =
                            scale2 * (((lowB3 & 0x0F) | (((highBits3 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i + 4] =
                            scale2 * (((lowB4 & 0x0F) | (((highBits4 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i + 5] =
                            scale2 * (((lowB5 & 0x0F) | (((highBits5 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i + 6] =
                            scale2 * (((lowB6 & 0x0F) | (((highBits6 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 32 + i + 7] =
                            scale2 * (((lowB7 & 0x0F) | (((highBits7 >>> 2) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i] =
                            scale3 * (((lowA0 >>> 4) | (((highBits0 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i + 1] =
                            scale3 * (((lowA1 >>> 4) | (((highBits1 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i + 2] =
                            scale3 * (((lowA2 >>> 4) | (((highBits2 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i + 3] =
                            scale3 * (((lowA3 >>> 4) | (((highBits3 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i + 4] =
                            scale3 * (((lowA4 >>> 4) | (((highBits4 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i + 5] =
                            scale3 * (((lowA5 >>> 4) | (((highBits5 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i + 6] =
                            scale3 * (((lowA6 >>> 4) | (((highBits6 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 64 + i + 7] =
                            scale3 * (((lowA7 >>> 4) | (((highBits7 >>> 4) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i] =
                            scale4 * (((lowB0 >>> 4) | (((highBits0 >>> 6) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i + 1] =
                            scale4 * (((lowB1 >>> 4) | (((highBits1 >>> 6) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i + 2] =
                            scale4 * (((lowB2 >>> 4) | (((highBits2 >>> 6) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i + 3] =
                            scale4 * (((lowB3 >>> 4) | (((highBits3 >>> 6) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i + 4] =
                            scale4 * (((lowB4 >>> 4) | (((highBits4 >>> 6) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i + 5] =
                            scale4 * (((lowB5 >>> 4) | (((highBits5 >>> 6) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i + 6] =
                            scale4 * (((lowB6 >>> 4) | (((highBits6 >>> 6) & 0x03) << 4)) - 32);
                    dst[outBase + 96 + i + 7] =
                            scale4 * (((lowB7 >>> 4) | (((highBits7 >>> 6) & 0x03) << 4)) - 32);
                }
            }
            lowBitsBase += 64;
            highBitsBase += 32;
            outBase += 128;
            scaleBase += 8;
        }
    }
}
