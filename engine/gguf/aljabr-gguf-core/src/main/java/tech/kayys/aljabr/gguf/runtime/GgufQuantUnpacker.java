package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.*;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Prepared-matrix quant unpacking routines.
 *
 * <p>This helper owns deterministic byte-layout expansion for GGUF quant blocks
 * so {@link GgufTensorOps} can keep matrix construction separate from format
 * unpack details.</p>
 * @author bhangun
 */
final class GgufQuantUnpacker {
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQuantUnpacker() {
    }

    static void unpackQ2KPreparedSuperBlock(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackQ2KPreparedLongPair(
                source.get(LE_LONG, sourceOffset),
                source.get(LE_LONG, sourceOffset + 16),
                quants,
                qBase,
                0);
        unpackQ2KPreparedLongPair(
                source.get(LE_LONG, sourceOffset + Long.BYTES),
                source.get(LE_LONG, sourceOffset + 16 + Long.BYTES),
                quants,
                qBase,
                8);
    }

    private static void unpackQ2KPreparedLongPair(
            long firstQuants,
            long secondQuants,
            byte[] quants,
            int qBase,
            int indexBase) {
        unpackQ2KPreparedByte(firstQuants, 0, quants, qBase, indexBase);
        unpackQ2KPreparedByte(firstQuants, 8, quants, qBase, indexBase + 1);
        unpackQ2KPreparedByte(firstQuants, 16, quants, qBase, indexBase + 2);
        unpackQ2KPreparedByte(firstQuants, 24, quants, qBase, indexBase + 3);
        unpackQ2KPreparedByte(firstQuants, 32, quants, qBase, indexBase + 4);
        unpackQ2KPreparedByte(firstQuants, 40, quants, qBase, indexBase + 5);
        unpackQ2KPreparedByte(firstQuants, 48, quants, qBase, indexBase + 6);
        unpackQ2KPreparedByte(firstQuants, 56, quants, qBase, indexBase + 7);
        unpackQ2KPreparedByte(secondQuants, 0, quants, qBase + 16, indexBase);
        unpackQ2KPreparedByte(secondQuants, 8, quants, qBase + 16, indexBase + 1);
        unpackQ2KPreparedByte(secondQuants, 16, quants, qBase + 16, indexBase + 2);
        unpackQ2KPreparedByte(secondQuants, 24, quants, qBase + 16, indexBase + 3);
        unpackQ2KPreparedByte(secondQuants, 32, quants, qBase + 16, indexBase + 4);
        unpackQ2KPreparedByte(secondQuants, 40, quants, qBase + 16, indexBase + 5);
        unpackQ2KPreparedByte(secondQuants, 48, quants, qBase + 16, indexBase + 6);
        unpackQ2KPreparedByte(secondQuants, 56, quants, qBase + 16, indexBase + 7);
    }

    private static void unpackQ2KPreparedByte(long packed, int shift, byte[] quants, int qBase, int index) {
        int quant = unsignedByte(packed, shift);
        quants[qBase + index] = (byte) (quant & 0x03);
        quants[qBase + 32 + index] = (byte) ((quant >>> 2) & 0x03);
        quants[qBase + 64 + index] = (byte) ((quant >>> 4) & 0x03);
        quants[qBase + 96 + index] = (byte) ((quant >>> 6) & 0x03);
    }

    static void unpackQ3KPreparedSuperBlock(
            MemorySegment source,
            long highMaskOffset,
            long sourceOffset,
            byte[] quants,
            int qBase,
            int mask0,
            int mask1,
            int mask2,
            int mask3) {
        unpackQ3KPreparedLongPair(
                source.get(LE_LONG, sourceOffset),
                source.get(LE_LONG, sourceOffset + 16),
                source.get(LE_LONG, highMaskOffset),
                source.get(LE_LONG, highMaskOffset + 16),
                quants,
                qBase,
                0,
                mask0,
                mask1,
                mask2,
                mask3);
        unpackQ3KPreparedLongPair(
                source.get(LE_LONG, sourceOffset + Long.BYTES),
                source.get(LE_LONG, sourceOffset + 16 + Long.BYTES),
                source.get(LE_LONG, highMaskOffset + Long.BYTES),
                source.get(LE_LONG, highMaskOffset + 16 + Long.BYTES),
                quants,
                qBase,
                8,
                mask0,
                mask1,
                mask2,
                mask3);
    }

    private static void unpackQ3KPreparedLongPair(
            long firstQuants,
            long secondQuants,
            long firstHighs,
            long secondHighs,
            byte[] quants,
            int qBase,
            int indexBase,
            int mask0,
            int mask1,
            int mask2,
            int mask3) {
        unpackQ3KPreparedByte(firstQuants, firstHighs, 0, quants, qBase, indexBase, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(firstQuants, firstHighs, 8, quants, qBase, indexBase + 1, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(firstQuants, firstHighs, 16, quants, qBase, indexBase + 2, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(firstQuants, firstHighs, 24, quants, qBase, indexBase + 3, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(firstQuants, firstHighs, 32, quants, qBase, indexBase + 4, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(firstQuants, firstHighs, 40, quants, qBase, indexBase + 5, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(firstQuants, firstHighs, 48, quants, qBase, indexBase + 6, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(firstQuants, firstHighs, 56, quants, qBase, indexBase + 7, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 0, quants, qBase + 16, indexBase, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 8, quants, qBase + 16, indexBase + 1, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 16, quants, qBase + 16, indexBase + 2, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 24, quants, qBase + 16, indexBase + 3, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 32, quants, qBase + 16, indexBase + 4, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 40, quants, qBase + 16, indexBase + 5, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 48, quants, qBase + 16, indexBase + 6, mask0, mask1, mask2, mask3);
        unpackQ3KPreparedByte(secondQuants, secondHighs, 56, quants, qBase + 16, indexBase + 7, mask0, mask1, mask2, mask3);
    }

    private static void unpackQ3KPreparedByte(
            long packed,
            long highs,
            int shift,
            byte[] quants,
            int qBase,
            int index,
            int mask0,
            int mask1,
            int mask2,
            int mask3) {
        int quant = unsignedByte(packed, shift);
        int highBits = unsignedByte(highs, shift);
        quants[qBase + index] = (byte) ((quant & 0x03) - q3KHighBias(highBits, mask0));
        quants[qBase + 32 + index] = (byte) (((quant >>> 2) & 0x03) - q3KHighBias(highBits, mask1));
        quants[qBase + 64 + index] = (byte) (((quant >>> 4) & 0x03) - q3KHighBias(highBits, mask2));
        quants[qBase + 96 + index] = (byte) (((quant >>> 6) & 0x03) - q3KHighBias(highBits, mask3));
    }

    static void unpackQ6KPreparedSuperBlock(
            MemorySegment source,
            long lowBitsOffset,
            long highBitsOffset,
            byte[] quants,
            int qBase) {
        unpackQ6KPreparedLong(
                source.get(LE_LONG, lowBitsOffset),
                source.get(LE_LONG, lowBitsOffset + Q4_0_BLOCK_SIZE),
                source.get(LE_LONG, highBitsOffset),
                quants,
                qBase,
                0);
        unpackQ6KPreparedLong(
                source.get(LE_LONG, lowBitsOffset + Long.BYTES),
                source.get(LE_LONG, lowBitsOffset + Q4_0_BLOCK_SIZE + Long.BYTES),
                source.get(LE_LONG, highBitsOffset + Long.BYTES),
                quants,
                qBase,
                8);
        unpackQ6KPreparedLong(
                source.get(LE_LONG, lowBitsOffset + 2L * Long.BYTES),
                source.get(LE_LONG, lowBitsOffset + Q4_0_BLOCK_SIZE + 2L * Long.BYTES),
                source.get(LE_LONG, highBitsOffset + 2L * Long.BYTES),
                quants,
                qBase,
                16);
        unpackQ6KPreparedLong(
                source.get(LE_LONG, lowBitsOffset + 3L * Long.BYTES),
                source.get(LE_LONG, lowBitsOffset + Q4_0_BLOCK_SIZE + 3L * Long.BYTES),
                source.get(LE_LONG, highBitsOffset + 3L * Long.BYTES),
                quants,
                qBase,
                24);
    }

    private static void unpackQ6KPreparedLong(
            long packedLowA,
            long packedLowB,
            long packedHighBits,
            byte[] quants,
            int qBase,
            int indexBase) {
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 0, quants, qBase, indexBase);
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 8, quants, qBase, indexBase + 1);
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 16, quants, qBase, indexBase + 2);
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 24, quants, qBase, indexBase + 3);
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 32, quants, qBase, indexBase + 4);
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 40, quants, qBase, indexBase + 5);
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 48, quants, qBase, indexBase + 6);
        unpackQ6KPreparedByte(packedLowA, packedLowB, packedHighBits, 56, quants, qBase, indexBase + 7);
    }

    private static void unpackQ6KPreparedByte(
            long packedLowA,
            long packedLowB,
            long packedHighBits,
            int shift,
            byte[] quants,
            int qBase,
            int index) {
        int lowA = unsignedByte(packedLowA, shift);
        int lowB = unsignedByte(packedLowB, shift);
        int highBits = unsignedByte(packedHighBits, shift);
        quants[qBase + index] = (byte) (((lowA & 0x0F) | ((highBits & 0x03) << 4)) - 32);
        quants[qBase + 32 + index] = (byte) (((lowB & 0x0F) | (((highBits >>> 2) & 0x03) << 4)) - 32);
        quants[qBase + 64 + index] = (byte) (((lowA >>> 4) | (((highBits >>> 4) & 0x03) << 4)) - 32);
        quants[qBase + 96 + index] = (byte) (((lowB >>> 4) | (((highBits >>> 6) & 0x03) << 4)) - 32);
    }

    static void unpackQ4KPreparedGroup(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackQ4KPreparedLong(source.get(LE_LONG, sourceOffset), quants, qBase, 0);
        unpackQ4KPreparedLong(source.get(LE_LONG, sourceOffset + Long.BYTES), quants, qBase, 8);
        unpackQ4KPreparedLong(source.get(LE_LONG, sourceOffset + 2L * Long.BYTES), quants, qBase, 16);
        unpackQ4KPreparedLong(source.get(LE_LONG, sourceOffset + 3L * Long.BYTES), quants, qBase, 24);
    }

    private static void unpackQ4KPreparedLong(long packed, byte[] quants, int qBase, int indexBase) {
        unpackQ4KPreparedByte(packed, 0, quants, qBase, indexBase);
        unpackQ4KPreparedByte(packed, 8, quants, qBase, indexBase + 1);
        unpackQ4KPreparedByte(packed, 16, quants, qBase, indexBase + 2);
        unpackQ4KPreparedByte(packed, 24, quants, qBase, indexBase + 3);
        unpackQ4KPreparedByte(packed, 32, quants, qBase, indexBase + 4);
        unpackQ4KPreparedByte(packed, 40, quants, qBase, indexBase + 5);
        unpackQ4KPreparedByte(packed, 48, quants, qBase, indexBase + 6);
        unpackQ4KPreparedByte(packed, 56, quants, qBase, indexBase + 7);
    }

    private static void unpackQ4KPreparedByte(long packed, int shift, byte[] quants, int qBase, int index) {
        int quant = unsignedByte(packed, shift);
        quants[qBase + index] = (byte) (quant & 0x0F);
        quants[qBase + Q4_0_BLOCK_SIZE + index] = (byte) (quant >>> 4);
    }

    static void unpackQ5KPreparedGroup(
            MemorySegment source,
            long sourceOffset,
            long highBitsOffset,
            int highMaskLow,
            int highMaskHigh,
            int highShiftLow,
            int highShiftHigh,
            byte[] quants,
            int qBase) {
        unpackQ5KPreparedLong(
                source.get(LE_LONG, sourceOffset),
                source.get(LE_LONG, highBitsOffset),
                highMaskLow,
                highMaskHigh,
                highShiftLow,
                highShiftHigh,
                quants,
                qBase,
                0);
        unpackQ5KPreparedLong(
                source.get(LE_LONG, sourceOffset + Long.BYTES),
                source.get(LE_LONG, highBitsOffset + Long.BYTES),
                highMaskLow,
                highMaskHigh,
                highShiftLow,
                highShiftHigh,
                quants,
                qBase,
                8);
        unpackQ5KPreparedLong(
                source.get(LE_LONG, sourceOffset + 2L * Long.BYTES),
                source.get(LE_LONG, highBitsOffset + 2L * Long.BYTES),
                highMaskLow,
                highMaskHigh,
                highShiftLow,
                highShiftHigh,
                quants,
                qBase,
                16);
        unpackQ5KPreparedLong(
                source.get(LE_LONG, sourceOffset + 3L * Long.BYTES),
                source.get(LE_LONG, highBitsOffset + 3L * Long.BYTES),
                highMaskLow,
                highMaskHigh,
                highShiftLow,
                highShiftHigh,
                quants,
                qBase,
                24);
    }

    private static void unpackQ5KPreparedLong(
            long packedQuants,
            long packedHighBits,
            int highMaskLow,
            int highMaskHigh,
            int highShiftLow,
            int highShiftHigh,
            byte[] quants,
            int qBase,
            int indexBase) {
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 0, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase);
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 8, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase + 1);
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 16, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase + 2);
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 24, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase + 3);
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 32, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase + 4);
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 40, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase + 5);
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 48, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase + 6);
        unpackQ5KPreparedByte(
                packedQuants, packedHighBits, 56, highMaskLow, highMaskHigh, highShiftLow, highShiftHigh,
                quants, qBase, indexBase + 7);
    }

    private static void unpackQ5KPreparedByte(
            long packedQuants,
            long packedHighBits,
            int shift,
            int highMaskLow,
            int highMaskHigh,
            int highShiftLow,
            int highShiftHigh,
            byte[] quants,
            int qBase,
            int index) {
        int quant = unsignedByte(packedQuants, shift);
        int highBits = unsignedByte(packedHighBits, shift);
        quants[qBase + index] = (byte) ((quant & 0x0F) | (((highBits & highMaskLow) >>> highShiftLow) << 4));
        quants[qBase + Q4_0_BLOCK_SIZE + index] =
                (byte) ((quant >>> 4) | (((highBits & highMaskHigh) >>> highShiftHigh) << 4));
    }

    static void unpackQ4_0Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        unpackPreparedQ4Block(source, sourceOffset, 8, quants, qBase);
    }

    static void unpackQ4_0Prepared(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackPreparedQ4Block(source, sourceOffset, 8, quants, qBase);
    }

    static void unpackQ4_1Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        unpackPreparedQ4Block(source, sourceOffset, 0, quants, qBase);
    }

    static void unpackQ4_1Prepared(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackPreparedQ4Block(source, sourceOffset, 0, quants, qBase);
    }

    static void unpackQ5_0Prepared(byte[] source, int sourceOffset, int highBits, byte[] quants, int qBase) {
        unpackPreparedQ5Block(source, sourceOffset, highBits, 16, quants, qBase);
    }

    static void unpackQ5_0Prepared(
            MemorySegment source,
            long sourceOffset,
            int highBits,
            byte[] quants,
            int qBase) {
        unpackPreparedQ5Block(source, sourceOffset, highBits, 16, quants, qBase);
    }

    static void unpackQ5_1Prepared(byte[] source, int sourceOffset, int highBits, byte[] quants, int qBase) {
        unpackPreparedQ5Block(source, sourceOffset, highBits, 0, quants, qBase);
    }

    static void unpackQ5_1Prepared(
            MemorySegment source,
            long sourceOffset,
            int highBits,
            byte[] quants,
            int qBase) {
        unpackPreparedQ5Block(source, sourceOffset, highBits, 0, quants, qBase);
    }

    private static void unpackPreparedQ4Block(
            byte[] source,
            int sourceOffset,
            int zeroPoint,
            byte[] quants,
            int qBase) {
        for (int i = 0; i < Q4_0_BLOCK_SIZE / 2; i++) {
            unpackPreparedQ4Byte(source[sourceOffset + i] & 0xFF, zeroPoint, quants, qBase, i);
        }
    }

    private static void unpackPreparedQ4Block(
            MemorySegment source,
            long sourceOffset,
            int zeroPoint,
            byte[] quants,
            int qBase) {
        unpackPreparedQ4Long(source.get(LE_LONG, sourceOffset), zeroPoint, quants, qBase, 0);
        unpackPreparedQ4Long(source.get(LE_LONG, sourceOffset + Long.BYTES), zeroPoint, quants, qBase, 8);
    }

    private static void unpackPreparedQ4Long(long packed, int zeroPoint, byte[] quants, int qBase, int indexBase) {
        unpackPreparedQ4Byte(packed, 0, zeroPoint, quants, qBase, indexBase);
        unpackPreparedQ4Byte(packed, 8, zeroPoint, quants, qBase, indexBase + 1);
        unpackPreparedQ4Byte(packed, 16, zeroPoint, quants, qBase, indexBase + 2);
        unpackPreparedQ4Byte(packed, 24, zeroPoint, quants, qBase, indexBase + 3);
        unpackPreparedQ4Byte(packed, 32, zeroPoint, quants, qBase, indexBase + 4);
        unpackPreparedQ4Byte(packed, 40, zeroPoint, quants, qBase, indexBase + 5);
        unpackPreparedQ4Byte(packed, 48, zeroPoint, quants, qBase, indexBase + 6);
        unpackPreparedQ4Byte(packed, 56, zeroPoint, quants, qBase, indexBase + 7);
    }

    private static void unpackPreparedQ4Byte(
            long packed,
            int shift,
            int zeroPoint,
            byte[] quants,
            int qBase,
            int index) {
        unpackPreparedQ4Byte(unsignedByte(packed, shift), zeroPoint, quants, qBase, index);
    }

    private static void unpackPreparedQ4Byte(int quant, int zeroPoint, byte[] quants, int qBase, int index) {
        quants[qBase + index] = (byte) ((quant & 0x0F) - zeroPoint);
        quants[qBase + Q4_0_BLOCK_SIZE / 2 + index] = (byte) ((quant >>> 4) - zeroPoint);
    }

    private static void unpackPreparedQ5Block(
            byte[] source,
            int sourceOffset,
            int highBits,
            int zeroPoint,
            byte[] quants,
            int qBase) {
        for (int i = 0; i < Q4_0_BLOCK_SIZE / 2; i++) {
            unpackPreparedQ5Byte(source[sourceOffset + i] & 0xFF, highBits, zeroPoint, quants, qBase, i);
        }
    }

    private static void unpackPreparedQ5Block(
            MemorySegment source,
            long sourceOffset,
            int highBits,
            int zeroPoint,
            byte[] quants,
            int qBase) {
        unpackPreparedQ5Long(source.get(LE_LONG, sourceOffset), highBits, zeroPoint, quants, qBase, 0);
        unpackPreparedQ5Long(source.get(LE_LONG, sourceOffset + Long.BYTES), highBits, zeroPoint, quants, qBase, 8);
    }

    private static void unpackPreparedQ5Long(
            long packed,
            int highBits,
            int zeroPoint,
            byte[] quants,
            int qBase,
            int indexBase) {
        unpackPreparedQ5Byte(packed, 0, highBits, zeroPoint, quants, qBase, indexBase);
        unpackPreparedQ5Byte(packed, 8, highBits, zeroPoint, quants, qBase, indexBase + 1);
        unpackPreparedQ5Byte(packed, 16, highBits, zeroPoint, quants, qBase, indexBase + 2);
        unpackPreparedQ5Byte(packed, 24, highBits, zeroPoint, quants, qBase, indexBase + 3);
        unpackPreparedQ5Byte(packed, 32, highBits, zeroPoint, quants, qBase, indexBase + 4);
        unpackPreparedQ5Byte(packed, 40, highBits, zeroPoint, quants, qBase, indexBase + 5);
        unpackPreparedQ5Byte(packed, 48, highBits, zeroPoint, quants, qBase, indexBase + 6);
        unpackPreparedQ5Byte(packed, 56, highBits, zeroPoint, quants, qBase, indexBase + 7);
    }

    private static void unpackPreparedQ5Byte(
            long packed,
            int shift,
            int highBits,
            int zeroPoint,
            byte[] quants,
            int qBase,
            int index) {
        unpackPreparedQ5Byte(unsignedByte(packed, shift), highBits, zeroPoint, quants, qBase, index);
    }

    private static void unpackPreparedQ5Byte(
            int quant,
            int highBits,
            int zeroPoint,
            byte[] quants,
            int qBase,
            int index) {
        int low = (quant & 0x0F) | (((highBits >>> index) & 1) << 4);
        int high = (quant >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
        quants[qBase + index] = (byte) (low - zeroPoint);
        quants[qBase + Q4_0_BLOCK_SIZE / 2 + index] = (byte) (high - zeroPoint);
    }

    static void unpackQ1_0Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        for (int packed = 0; packed < Q1_0_BLOCK_SIZE / 8; packed++) {
            int mask = source[sourceOffset + packed] & 0xFF;
            copyQ1_0Signs(mask, quants, qBase + packed * 8);
        }
    }

    static void unpackQ1_0Prepared(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackQ1_0PreparedLong(source.get(LE_LONG, sourceOffset), quants, qBase);
        unpackQ1_0PreparedLong(source.get(LE_LONG, sourceOffset + Long.BYTES), quants, qBase + 64);
    }

    private static void unpackQ1_0PreparedLong(long masks, byte[] quants, int outBase) {
        unpackQ1_0PreparedByte(masks, 0, quants, outBase);
        unpackQ1_0PreparedByte(masks, 8, quants, outBase + 8);
        unpackQ1_0PreparedByte(masks, 16, quants, outBase + 16);
        unpackQ1_0PreparedByte(masks, 24, quants, outBase + 24);
        unpackQ1_0PreparedByte(masks, 32, quants, outBase + 32);
        unpackQ1_0PreparedByte(masks, 40, quants, outBase + 40);
        unpackQ1_0PreparedByte(masks, 48, quants, outBase + 48);
        unpackQ1_0PreparedByte(masks, 56, quants, outBase + 56);
    }

    private static void unpackQ1_0PreparedByte(long masks, int shift, byte[] quants, int outBase) {
        int mask = unsignedByte(masks, shift);
        copyQ1_0Signs(mask, quants, outBase);
    }

    private static void copyQ1_0Signs(int mask, byte[] quants, int outBase) {
        int signBase = q1_0SignBase(mask);
        quants[outBase] = Q1_0_SIGNS[signBase];
        quants[outBase + 1] = Q1_0_SIGNS[signBase + 1];
        quants[outBase + 2] = Q1_0_SIGNS[signBase + 2];
        quants[outBase + 3] = Q1_0_SIGNS[signBase + 3];
        quants[outBase + 4] = Q1_0_SIGNS[signBase + 4];
        quants[outBase + 5] = Q1_0_SIGNS[signBase + 5];
        quants[outBase + 6] = Q1_0_SIGNS[signBase + 6];
        quants[outBase + 7] = Q1_0_SIGNS[signBase + 7];
    }

    static void unpackTQ1_0Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        int out = qBase;
        for (int group = 0; group < TQ1_0_QUANT_BYTES - TQ1_0_QUANT_BYTES % 32; group += 32) {
            int groupBase = out;
            for (int i = 0; i < 32; i++) {
                int packed = source[sourceOffset + group + i] & 0xFF;
                int tritBase = tq1_0TritBase(packed);
                quants[groupBase + i] = TQ1_0_TRITS[tritBase];
                quants[groupBase + 32 + i] = TQ1_0_TRITS[tritBase + 1];
                quants[groupBase + 64 + i] = TQ1_0_TRITS[tritBase + 2];
                quants[groupBase + 96 + i] = TQ1_0_TRITS[tritBase + 3];
                quants[groupBase + 128 + i] = TQ1_0_TRITS[tritBase + 4];
            }
            out += 160;
        }
        for (int group = TQ1_0_QUANT_BYTES - TQ1_0_QUANT_BYTES % 32;
                group < TQ1_0_QUANT_BYTES;
                group += 16) {
            int groupBase = out;
            for (int i = 0; i < 16; i++) {
                int packed = source[sourceOffset + group + i] & 0xFF;
                int tritBase = tq1_0TritBase(packed);
                quants[groupBase + i] = TQ1_0_TRITS[tritBase];
                quants[groupBase + 16 + i] = TQ1_0_TRITS[tritBase + 1];
                quants[groupBase + 32 + i] = TQ1_0_TRITS[tritBase + 2];
                quants[groupBase + 48 + i] = TQ1_0_TRITS[tritBase + 3];
                quants[groupBase + 64 + i] = TQ1_0_TRITS[tritBase + 4];
            }
            out += 80;
        }
        int highOffset = sourceOffset + TQ1_0_QUANT_BYTES;
        int highBase = out;
        for (int i = 0; i < TQ1_0_HIGH_BYTES; i++) {
            int packed = source[highOffset + i] & 0xFF;
            int tritBase = tq1_0TritBase(packed);
            quants[highBase + i] = TQ1_0_TRITS[tritBase];
            quants[highBase + 4 + i] = TQ1_0_TRITS[tritBase + 1];
            quants[highBase + 8 + i] = TQ1_0_TRITS[tritBase + 2];
            quants[highBase + 12 + i] = TQ1_0_TRITS[tritBase + 3];
        }
    }

    static void unpackTQ1_0Prepared(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackTQ1_0PreparedGroup32(source, sourceOffset, quants, qBase);
        unpackTQ1_0PreparedGroup16(source, sourceOffset + 32, quants, qBase + 160);
        long highOffset = sourceOffset + TQ1_0_QUANT_BYTES;
        int highBase = qBase + 240;
        int highPacked = source.get(LE_INT, highOffset);
        unpackTQ1_0PreparedHighByte(highPacked, 0, quants, highBase, 0);
        unpackTQ1_0PreparedHighByte(highPacked, 8, quants, highBase, 1);
        unpackTQ1_0PreparedHighByte(highPacked, 16, quants, highBase, 2);
        unpackTQ1_0PreparedHighByte(highPacked, 24, quants, highBase, 3);
    }

    private static void unpackTQ1_0PreparedGroup32(
            MemorySegment source,
            long groupOffset,
            byte[] quants,
            int groupBase) {
        unpackTQ1_0PreparedPacked32(source.get(LE_LONG, groupOffset), quants, groupBase, 0);
        unpackTQ1_0PreparedPacked32(source.get(LE_LONG, groupOffset + Long.BYTES), quants, groupBase, 8);
        unpackTQ1_0PreparedPacked32(source.get(LE_LONG, groupOffset + 2L * Long.BYTES), quants, groupBase, 16);
        unpackTQ1_0PreparedPacked32(source.get(LE_LONG, groupOffset + 3L * Long.BYTES), quants, groupBase, 24);
    }

    private static void unpackTQ1_0PreparedGroup16(
            MemorySegment source,
            long groupOffset,
            byte[] quants,
            int groupBase) {
        unpackTQ1_0PreparedPacked16(source.get(LE_LONG, groupOffset), quants, groupBase, 0);
        unpackTQ1_0PreparedPacked16(source.get(LE_LONG, groupOffset + Long.BYTES), quants, groupBase, 8);
    }

    private static void unpackTQ1_0PreparedPacked32(long packed, byte[] quants, int groupBase, int indexBase) {
        unpackTQ1_0PreparedByte32(packed, 0, quants, groupBase, indexBase);
        unpackTQ1_0PreparedByte32(packed, 8, quants, groupBase, indexBase + 1);
        unpackTQ1_0PreparedByte32(packed, 16, quants, groupBase, indexBase + 2);
        unpackTQ1_0PreparedByte32(packed, 24, quants, groupBase, indexBase + 3);
        unpackTQ1_0PreparedByte32(packed, 32, quants, groupBase, indexBase + 4);
        unpackTQ1_0PreparedByte32(packed, 40, quants, groupBase, indexBase + 5);
        unpackTQ1_0PreparedByte32(packed, 48, quants, groupBase, indexBase + 6);
        unpackTQ1_0PreparedByte32(packed, 56, quants, groupBase, indexBase + 7);
    }

    private static void unpackTQ1_0PreparedPacked16(long packed, byte[] quants, int groupBase, int indexBase) {
        unpackTQ1_0PreparedByte16(packed, 0, quants, groupBase, indexBase);
        unpackTQ1_0PreparedByte16(packed, 8, quants, groupBase, indexBase + 1);
        unpackTQ1_0PreparedByte16(packed, 16, quants, groupBase, indexBase + 2);
        unpackTQ1_0PreparedByte16(packed, 24, quants, groupBase, indexBase + 3);
        unpackTQ1_0PreparedByte16(packed, 32, quants, groupBase, indexBase + 4);
        unpackTQ1_0PreparedByte16(packed, 40, quants, groupBase, indexBase + 5);
        unpackTQ1_0PreparedByte16(packed, 48, quants, groupBase, indexBase + 6);
        unpackTQ1_0PreparedByte16(packed, 56, quants, groupBase, indexBase + 7);
    }

    private static void unpackTQ1_0PreparedByte32(
            long packed,
            int shift,
            byte[] quants,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        quants[groupBase + index] = TQ1_0_TRITS[tritBase];
        quants[groupBase + 32 + index] = TQ1_0_TRITS[tritBase + 1];
        quants[groupBase + 64 + index] = TQ1_0_TRITS[tritBase + 2];
        quants[groupBase + 96 + index] = TQ1_0_TRITS[tritBase + 3];
        quants[groupBase + 128 + index] = TQ1_0_TRITS[tritBase + 4];
    }

    private static void unpackTQ1_0PreparedByte16(
            long packed,
            int shift,
            byte[] quants,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        quants[groupBase + index] = TQ1_0_TRITS[tritBase];
        quants[groupBase + 16 + index] = TQ1_0_TRITS[tritBase + 1];
        quants[groupBase + 32 + index] = TQ1_0_TRITS[tritBase + 2];
        quants[groupBase + 48 + index] = TQ1_0_TRITS[tritBase + 3];
        quants[groupBase + 64 + index] = TQ1_0_TRITS[tritBase + 4];
    }

    private static void unpackTQ1_0PreparedHighByte(
            int packed,
            int shift,
            byte[] quants,
            int highBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        quants[highBase + index] = TQ1_0_TRITS[tritBase];
        quants[highBase + 4 + index] = TQ1_0_TRITS[tritBase + 1];
        quants[highBase + 8 + index] = TQ1_0_TRITS[tritBase + 2];
        quants[highBase + 12 + index] = TQ1_0_TRITS[tritBase + 3];
    }

    static void unpackTQ2_0Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        for (int group = 0; group < TQ2_0_QUANT_BYTES; group += 32) {
            int groupBase = qBase + group * 4;
            for (int i = 0; i < 32; i++) {
                int quant = source[sourceOffset + group + i] & 0xFF;
                int laneBase = tq2_0LaneBase(quant);
                quants[groupBase + i] = TQ2_0_LANES[laneBase];
                quants[groupBase + 32 + i] = TQ2_0_LANES[laneBase + 1];
                quants[groupBase + 64 + i] = TQ2_0_LANES[laneBase + 2];
                quants[groupBase + 96 + i] = TQ2_0_LANES[laneBase + 3];
            }
        }
    }

    static void unpackTQ2_0Prepared(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackTQ2_0PreparedGroup(source, sourceOffset, quants, qBase);
        unpackTQ2_0PreparedGroup(source, sourceOffset + 32, quants, qBase + 128);
    }

    private static void unpackTQ2_0PreparedGroup(MemorySegment source, long groupOffset, byte[] quants, int groupBase) {
        unpackTQ2_0PreparedPacked(source.get(LE_LONG, groupOffset), quants, groupBase, 0);
        unpackTQ2_0PreparedPacked(source.get(LE_LONG, groupOffset + Long.BYTES), quants, groupBase, 8);
        unpackTQ2_0PreparedPacked(source.get(LE_LONG, groupOffset + 2L * Long.BYTES), quants, groupBase, 16);
        unpackTQ2_0PreparedPacked(source.get(LE_LONG, groupOffset + 3L * Long.BYTES), quants, groupBase, 24);
    }

    private static void unpackTQ2_0PreparedPacked(long packed, byte[] quants, int groupBase, int indexBase) {
        unpackTQ2_0PreparedByte(packed, 0, quants, groupBase, indexBase);
        unpackTQ2_0PreparedByte(packed, 8, quants, groupBase, indexBase + 1);
        unpackTQ2_0PreparedByte(packed, 16, quants, groupBase, indexBase + 2);
        unpackTQ2_0PreparedByte(packed, 24, quants, groupBase, indexBase + 3);
        unpackTQ2_0PreparedByte(packed, 32, quants, groupBase, indexBase + 4);
        unpackTQ2_0PreparedByte(packed, 40, quants, groupBase, indexBase + 5);
        unpackTQ2_0PreparedByte(packed, 48, quants, groupBase, indexBase + 6);
        unpackTQ2_0PreparedByte(packed, 56, quants, groupBase, indexBase + 7);
    }

    private static void unpackTQ2_0PreparedByte(long packed, int shift, byte[] quants, int groupBase, int index) {
        int quant = unsignedByte(packed, shift);
        int laneBase = tq2_0LaneBase(quant);
        quants[groupBase + index] = TQ2_0_LANES[laneBase];
        quants[groupBase + 32 + index] = TQ2_0_LANES[laneBase + 1];
        quants[groupBase + 64 + index] = TQ2_0_LANES[laneBase + 2];
        quants[groupBase + 96 + index] = TQ2_0_LANES[laneBase + 3];
    }

    static void copyPreparedQuants(
            MemorySegment source,
            long sourceOffset,
            byte[] quants,
            int qBase,
            int length) {
        MemorySegment.copy(source, ValueLayout.JAVA_BYTE, sourceOffset, quants, qBase, length);
    }

    static void unpackMXFP4Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        unpackPreparedNibble32(source, sourceOffset, MXFP4_NIBBLE_PAIRS, quants, qBase);
    }

    static void unpackMXFP4Prepared(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackPreparedNibble32(source, sourceOffset, MXFP4_NIBBLE_PAIRS, quants, qBase);
    }

    static void unpackNVFP4SubBlockPrepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        unpackPreparedNibble16(source, sourceOffset, MXFP4_NIBBLE_PAIRS, quants, qBase);
    }

    static void unpackNVFP4SubBlockPrepared(
            MemorySegment source,
            long sourceOffset,
            byte[] quants,
            int qBase) {
        unpackPreparedNibble16(source, sourceOffset, MXFP4_NIBBLE_PAIRS, quants, qBase);
    }

    static void unpackIQ4NLPrepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        unpackPreparedNibble32(source, sourceOffset, IQ4_NL_NIBBLE_PAIRS, quants, qBase);
    }

    static void unpackIQ4NLPrepared(MemorySegment source, long sourceOffset, byte[] quants, int qBase) {
        unpackPreparedNibble32(source, sourceOffset, IQ4_NL_NIBBLE_PAIRS, quants, qBase);
    }

    private static void unpackPreparedNibble16(
            byte[] source,
            int sourceOffset,
            byte[] pairValues,
            byte[] quants,
            int qBase) {
        unpackPreparedNibbleByte8(source[sourceOffset] & 0xFF, pairValues, quants, qBase, 0);
        unpackPreparedNibbleByte8(source[sourceOffset + 1] & 0xFF, pairValues, quants, qBase, 1);
        unpackPreparedNibbleByte8(source[sourceOffset + 2] & 0xFF, pairValues, quants, qBase, 2);
        unpackPreparedNibbleByte8(source[sourceOffset + 3] & 0xFF, pairValues, quants, qBase, 3);
        unpackPreparedNibbleByte8(source[sourceOffset + 4] & 0xFF, pairValues, quants, qBase, 4);
        unpackPreparedNibbleByte8(source[sourceOffset + 5] & 0xFF, pairValues, quants, qBase, 5);
        unpackPreparedNibbleByte8(source[sourceOffset + 6] & 0xFF, pairValues, quants, qBase, 6);
        unpackPreparedNibbleByte8(source[sourceOffset + 7] & 0xFF, pairValues, quants, qBase, 7);
    }

    private static void unpackPreparedNibble16(
            MemorySegment source,
            long sourceOffset,
            byte[] pairValues,
            byte[] quants,
            int qBase) {
        long packed = source.get(LE_LONG, sourceOffset);
        unpackPreparedNibbleByte8(packed, 0, pairValues, quants, qBase, 0);
        unpackPreparedNibbleByte8(packed, 8, pairValues, quants, qBase, 1);
        unpackPreparedNibbleByte8(packed, 16, pairValues, quants, qBase, 2);
        unpackPreparedNibbleByte8(packed, 24, pairValues, quants, qBase, 3);
        unpackPreparedNibbleByte8(packed, 32, pairValues, quants, qBase, 4);
        unpackPreparedNibbleByte8(packed, 40, pairValues, quants, qBase, 5);
        unpackPreparedNibbleByte8(packed, 48, pairValues, quants, qBase, 6);
        unpackPreparedNibbleByte8(packed, 56, pairValues, quants, qBase, 7);
    }

    private static void unpackPreparedNibble32(
            byte[] source,
            int sourceOffset,
            byte[] pairValues,
            byte[] quants,
            int qBase) {
        unpackPreparedNibbleByte16(source[sourceOffset] & 0xFF, pairValues, quants, qBase, 0);
        unpackPreparedNibbleByte16(source[sourceOffset + 1] & 0xFF, pairValues, quants, qBase, 1);
        unpackPreparedNibbleByte16(source[sourceOffset + 2] & 0xFF, pairValues, quants, qBase, 2);
        unpackPreparedNibbleByte16(source[sourceOffset + 3] & 0xFF, pairValues, quants, qBase, 3);
        unpackPreparedNibbleByte16(source[sourceOffset + 4] & 0xFF, pairValues, quants, qBase, 4);
        unpackPreparedNibbleByte16(source[sourceOffset + 5] & 0xFF, pairValues, quants, qBase, 5);
        unpackPreparedNibbleByte16(source[sourceOffset + 6] & 0xFF, pairValues, quants, qBase, 6);
        unpackPreparedNibbleByte16(source[sourceOffset + 7] & 0xFF, pairValues, quants, qBase, 7);
        unpackPreparedNibbleByte16(source[sourceOffset + 8] & 0xFF, pairValues, quants, qBase, 8);
        unpackPreparedNibbleByte16(source[sourceOffset + 9] & 0xFF, pairValues, quants, qBase, 9);
        unpackPreparedNibbleByte16(source[sourceOffset + 10] & 0xFF, pairValues, quants, qBase, 10);
        unpackPreparedNibbleByte16(source[sourceOffset + 11] & 0xFF, pairValues, quants, qBase, 11);
        unpackPreparedNibbleByte16(source[sourceOffset + 12] & 0xFF, pairValues, quants, qBase, 12);
        unpackPreparedNibbleByte16(source[sourceOffset + 13] & 0xFF, pairValues, quants, qBase, 13);
        unpackPreparedNibbleByte16(source[sourceOffset + 14] & 0xFF, pairValues, quants, qBase, 14);
        unpackPreparedNibbleByte16(source[sourceOffset + 15] & 0xFF, pairValues, quants, qBase, 15);
    }

    private static void unpackPreparedNibble32(
            MemorySegment source,
            long sourceOffset,
            byte[] pairValues,
            byte[] quants,
            int qBase) {
        long first = source.get(LE_LONG, sourceOffset);
        long second = source.get(LE_LONG, sourceOffset + Long.BYTES);
        unpackPreparedNibbleByte16(first, 0, pairValues, quants, qBase, 0);
        unpackPreparedNibbleByte16(first, 8, pairValues, quants, qBase, 1);
        unpackPreparedNibbleByte16(first, 16, pairValues, quants, qBase, 2);
        unpackPreparedNibbleByte16(first, 24, pairValues, quants, qBase, 3);
        unpackPreparedNibbleByte16(first, 32, pairValues, quants, qBase, 4);
        unpackPreparedNibbleByte16(first, 40, pairValues, quants, qBase, 5);
        unpackPreparedNibbleByte16(first, 48, pairValues, quants, qBase, 6);
        unpackPreparedNibbleByte16(first, 56, pairValues, quants, qBase, 7);
        unpackPreparedNibbleByte16(second, 0, pairValues, quants, qBase, 8);
        unpackPreparedNibbleByte16(second, 8, pairValues, quants, qBase, 9);
        unpackPreparedNibbleByte16(second, 16, pairValues, quants, qBase, 10);
        unpackPreparedNibbleByte16(second, 24, pairValues, quants, qBase, 11);
        unpackPreparedNibbleByte16(second, 32, pairValues, quants, qBase, 12);
        unpackPreparedNibbleByte16(second, 40, pairValues, quants, qBase, 13);
        unpackPreparedNibbleByte16(second, 48, pairValues, quants, qBase, 14);
        unpackPreparedNibbleByte16(second, 56, pairValues, quants, qBase, 15);
    }

    private static void unpackPreparedNibbleByte8(
            long packed,
            int shift,
            byte[] pairValues,
            byte[] quants,
            int qBase,
            int index) {
        unpackPreparedNibbleByte8(unsignedByte(packed, shift), pairValues, quants, qBase, index);
    }

    private static void unpackPreparedNibbleByte8(
            int quant,
            byte[] pairValues,
            byte[] quants,
            int qBase,
            int index) {
        int pairBase = nibblePairBase(quant);
        quants[qBase + index] = pairValues[pairBase];
        quants[qBase + 8 + index] = pairValues[pairBase + 1];
    }

    private static void unpackPreparedNibbleByte16(
            long packed,
            int shift,
            byte[] pairValues,
            byte[] quants,
            int qBase,
            int index) {
        unpackPreparedNibbleByte16(unsignedByte(packed, shift), pairValues, quants, qBase, index);
    }

    private static void unpackPreparedNibbleByte16(
            int quant,
            byte[] pairValues,
            byte[] quants,
            int qBase,
            int index) {
        int pairBase = nibblePairBase(quant);
        quants[qBase + index] = pairValues[pairBase];
        quants[qBase + 16 + index] = pairValues[pairBase + 1];
    }
}
