package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.*;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Block-level dequantizers for compact GGUF formats.
 *
 * <p>This class owns single-block expansion for non-K and nibble-family
 * formats. Larger K-family block decoders can move here in smaller follow-up
 * slices once their call sites are isolated.</p>
 * @author bhangun
 */
final class GgufBlockDequantizer {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufBlockDequantizer() {
    }

    static void dequantizeQ4_0Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        long quantsOffset = blockOffset + 2;
        dequantizeQ4Block(segment, quantsOffset, d, 0.0f, 8, dst, dstOffset);
    }

    static void dequantizeQ1_0Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        long bitsOffset = blockOffset + 2;
        dequantizePackedQ1Long(segment.get(LE_LONG, bitsOffset), d, dst, dstOffset);
        dequantizePackedQ1Long(segment.get(LE_LONG, bitsOffset + Long.BYTES), d, dst, dstOffset + 64);
    }

    static void dequantizeTQ1_0Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + TQ1_0_SCALE_OFFSET));
        dequantizeTQ1_0Group32(segment, blockOffset, d, dst, dstOffset);
        dequantizeTQ1_0Group16(segment, blockOffset + 32, d, dst, dstOffset + 160);
        long highOffset = blockOffset + TQ1_0_QUANT_BYTES;
        int highBase = dstOffset + 240;
        int highPacked = segment.get(LE_INT, highOffset);
        dequantizePackedTQ1_0HighByte(highPacked, 0, d, dst, highBase, 0);
        dequantizePackedTQ1_0HighByte(highPacked, 8, d, dst, highBase, 1);
        dequantizePackedTQ1_0HighByte(highPacked, 16, d, dst, highBase, 2);
        dequantizePackedTQ1_0HighByte(highPacked, 24, d, dst, highBase, 3);
    }

    private static void dequantizeTQ1_0Group32(
            MemorySegment segment,
            long groupOffset,
            float scale,
            float[] dst,
            int groupBase) {
        dequantizePackedTQ1_0Long32(segment.get(LE_LONG, groupOffset), scale, dst, groupBase, 0);
        dequantizePackedTQ1_0Long32(segment.get(LE_LONG, groupOffset + Long.BYTES), scale, dst, groupBase, 8);
        dequantizePackedTQ1_0Long32(segment.get(LE_LONG, groupOffset + 2L * Long.BYTES), scale, dst, groupBase, 16);
        dequantizePackedTQ1_0Long32(segment.get(LE_LONG, groupOffset + 3L * Long.BYTES), scale, dst, groupBase, 24);
    }

    private static void dequantizeTQ1_0Group16(
            MemorySegment segment,
            long groupOffset,
            float scale,
            float[] dst,
            int groupBase) {
        dequantizePackedTQ1_0Long16(segment.get(LE_LONG, groupOffset), scale, dst, groupBase, 0);
        dequantizePackedTQ1_0Long16(segment.get(LE_LONG, groupOffset + Long.BYTES), scale, dst, groupBase, 8);
    }

    private static void dequantizePackedTQ1_0Long32(
            long packed,
            float scale,
            float[] dst,
            int groupBase,
            int indexBase) {
        dequantizePackedTQ1_0Byte32(packed, 0, scale, dst, groupBase, indexBase);
        dequantizePackedTQ1_0Byte32(packed, 8, scale, dst, groupBase, indexBase + 1);
        dequantizePackedTQ1_0Byte32(packed, 16, scale, dst, groupBase, indexBase + 2);
        dequantizePackedTQ1_0Byte32(packed, 24, scale, dst, groupBase, indexBase + 3);
        dequantizePackedTQ1_0Byte32(packed, 32, scale, dst, groupBase, indexBase + 4);
        dequantizePackedTQ1_0Byte32(packed, 40, scale, dst, groupBase, indexBase + 5);
        dequantizePackedTQ1_0Byte32(packed, 48, scale, dst, groupBase, indexBase + 6);
        dequantizePackedTQ1_0Byte32(packed, 56, scale, dst, groupBase, indexBase + 7);
    }

    private static void dequantizePackedTQ1_0Long16(
            long packed,
            float scale,
            float[] dst,
            int groupBase,
            int indexBase) {
        dequantizePackedTQ1_0Byte16(packed, 0, scale, dst, groupBase, indexBase);
        dequantizePackedTQ1_0Byte16(packed, 8, scale, dst, groupBase, indexBase + 1);
        dequantizePackedTQ1_0Byte16(packed, 16, scale, dst, groupBase, indexBase + 2);
        dequantizePackedTQ1_0Byte16(packed, 24, scale, dst, groupBase, indexBase + 3);
        dequantizePackedTQ1_0Byte16(packed, 32, scale, dst, groupBase, indexBase + 4);
        dequantizePackedTQ1_0Byte16(packed, 40, scale, dst, groupBase, indexBase + 5);
        dequantizePackedTQ1_0Byte16(packed, 48, scale, dst, groupBase, indexBase + 6);
        dequantizePackedTQ1_0Byte16(packed, 56, scale, dst, groupBase, indexBase + 7);
    }

    private static void dequantizePackedTQ1_0Byte32(
            long packed,
            int shift,
            float scale,
            float[] dst,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        dst[groupBase + index] = TQ1_0_TRITS[tritBase] * scale;
        dst[groupBase + 32 + index] = TQ1_0_TRITS[tritBase + 1] * scale;
        dst[groupBase + 64 + index] = TQ1_0_TRITS[tritBase + 2] * scale;
        dst[groupBase + 96 + index] = TQ1_0_TRITS[tritBase + 3] * scale;
        dst[groupBase + 128 + index] = TQ1_0_TRITS[tritBase + 4] * scale;
    }

    private static void dequantizePackedTQ1_0Byte16(
            long packed,
            int shift,
            float scale,
            float[] dst,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        dst[groupBase + index] = TQ1_0_TRITS[tritBase] * scale;
        dst[groupBase + 16 + index] = TQ1_0_TRITS[tritBase + 1] * scale;
        dst[groupBase + 32 + index] = TQ1_0_TRITS[tritBase + 2] * scale;
        dst[groupBase + 48 + index] = TQ1_0_TRITS[tritBase + 3] * scale;
        dst[groupBase + 64 + index] = TQ1_0_TRITS[tritBase + 4] * scale;
    }

    private static void dequantizePackedTQ1_0HighByte(
            int packed,
            int shift,
            float scale,
            float[] dst,
            int highBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        dst[highBase + index] = TQ1_0_TRITS[tritBase] * scale;
        dst[highBase + 4 + index] = TQ1_0_TRITS[tritBase + 1] * scale;
        dst[highBase + 8 + index] = TQ1_0_TRITS[tritBase + 2] * scale;
        dst[highBase + 12 + index] = TQ1_0_TRITS[tritBase + 3] * scale;
    }

    static void dequantizeTQ2_0Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + TQ2_0_QUANT_BYTES));
        dequantizeTQ2_0Group(segment, blockOffset, d, dst, dstOffset);
        dequantizeTQ2_0Group(segment, blockOffset + 32, d, dst, dstOffset + 128);
    }

    private static void dequantizeTQ2_0Group(
            MemorySegment segment,
            long groupOffset,
            float scale,
            float[] dst,
            int groupBase) {
        dequantizePackedTQ2_0Long(segment.get(LE_LONG, groupOffset), scale, dst, groupBase, 0);
        dequantizePackedTQ2_0Long(segment.get(LE_LONG, groupOffset + Long.BYTES), scale, dst, groupBase, 8);
        dequantizePackedTQ2_0Long(segment.get(LE_LONG, groupOffset + 2L * Long.BYTES), scale, dst, groupBase, 16);
        dequantizePackedTQ2_0Long(segment.get(LE_LONG, groupOffset + 3L * Long.BYTES), scale, dst, groupBase, 24);
    }

    private static void dequantizePackedTQ2_0Long(
            long packed,
            float scale,
            float[] dst,
            int groupBase,
            int indexBase) {
        dequantizePackedTQ2_0Byte(packed, 0, scale, dst, groupBase, indexBase);
        dequantizePackedTQ2_0Byte(packed, 8, scale, dst, groupBase, indexBase + 1);
        dequantizePackedTQ2_0Byte(packed, 16, scale, dst, groupBase, indexBase + 2);
        dequantizePackedTQ2_0Byte(packed, 24, scale, dst, groupBase, indexBase + 3);
        dequantizePackedTQ2_0Byte(packed, 32, scale, dst, groupBase, indexBase + 4);
        dequantizePackedTQ2_0Byte(packed, 40, scale, dst, groupBase, indexBase + 5);
        dequantizePackedTQ2_0Byte(packed, 48, scale, dst, groupBase, indexBase + 6);
        dequantizePackedTQ2_0Byte(packed, 56, scale, dst, groupBase, indexBase + 7);
    }

    private static void dequantizePackedTQ2_0Byte(
            long packed,
            int shift,
            float scale,
            float[] dst,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int laneBase = tq2_0LaneBase(quant);
        dst[groupBase + index] = TQ2_0_LANES[laneBase] * scale;
        dst[groupBase + 32 + index] = TQ2_0_LANES[laneBase + 1] * scale;
        dst[groupBase + 64 + index] = TQ2_0_LANES[laneBase + 2] * scale;
        dst[groupBase + 96 + index] = TQ2_0_LANES[laneBase + 3] * scale;
    }

    static void dequantizeQ4_1Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        long quantsOffset = blockOffset + 4;
        dequantizeQ4Block(segment, quantsOffset, d, m, 0, dst, dstOffset);
    }

    static void dequantizeQ5_0Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        int highBits = segment.get(LE_INT, blockOffset + 2);
        long quantsOffset = blockOffset + 6;
        dequantizeQ5Block(segment, quantsOffset, highBits, d, 0.0f, 16, dst, dstOffset);
    }

    static void dequantizeQ5_1Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        int highBits = segment.get(LE_INT, blockOffset + 4);
        long quantsOffset = blockOffset + 8;
        dequantizeQ5Block(segment, quantsOffset, highBits, d, m, 0, dst, dstOffset);
    }

    static void dequantizeQ8Block(
            MemorySegment segment,
            long quantsOffset,
            float scale,
            float[] dst,
            int dstOffset) {
        dequantizePackedQ8Long(segment.get(LE_LONG, quantsOffset), scale, dst, dstOffset);
        dequantizePackedQ8Long(segment.get(LE_LONG, quantsOffset + Long.BYTES), scale, dst, dstOffset + 8);
        dequantizePackedQ8Long(segment.get(LE_LONG, quantsOffset + 2L * Long.BYTES), scale, dst, dstOffset + 16);
        dequantizePackedQ8Long(segment.get(LE_LONG, quantsOffset + 3L * Long.BYTES), scale, dst, dstOffset + 24);
    }

    static void dequantizeQ8KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = segment.get(LE_FLOAT, blockOffset);
        long quantsOffset = blockOffset + Float.BYTES;
        dequantizeQ8Block(segment, quantsOffset, d, dst, dstOffset);
        dequantizeQ8Block(segment, quantsOffset + 32, d, dst, dstOffset + 32);
        dequantizeQ8Block(segment, quantsOffset + 64, d, dst, dstOffset + 64);
        dequantizeQ8Block(segment, quantsOffset + 96, d, dst, dstOffset + 96);
        dequantizeQ8Block(segment, quantsOffset + 128, d, dst, dstOffset + 128);
        dequantizeQ8Block(segment, quantsOffset + 160, d, dst, dstOffset + 160);
        dequantizeQ8Block(segment, quantsOffset + 192, d, dst, dstOffset + 192);
        dequantizeQ8Block(segment, quantsOffset + 224, d, dst, dstOffset + 224);
    }

    private static void dequantizePackedQ8Long(long packed, float scale, float[] dst, int dstOffset) {
        dst[dstOffset] = signedByte(packed, 0) * scale;
        dst[dstOffset + 1] = signedByte(packed, 8) * scale;
        dst[dstOffset + 2] = signedByte(packed, 16) * scale;
        dst[dstOffset + 3] = signedByte(packed, 24) * scale;
        dst[dstOffset + 4] = signedByte(packed, 32) * scale;
        dst[dstOffset + 5] = signedByte(packed, 40) * scale;
        dst[dstOffset + 6] = signedByte(packed, 48) * scale;
        dst[dstOffset + 7] = signedByte(packed, 56) * scale;
    }

    static void dequantizeIQ4NLBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        long quantsOffset = blockOffset + 2;
        dequantizeNibble32(segment, quantsOffset, d, IQ4_NL_NIBBLE_PAIRS, dst, dstOffset);
    }

    static void dequantizeMXFP4Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = e8m0ToF32Half(u8(segment, blockOffset));
        long quantsOffset = blockOffset + 1;
        dequantizeNibble32(segment, quantsOffset, d, MXFP4_NIBBLE_PAIRS, dst, dstOffset);
    }

    static void dequantizeNVFP4Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        long scalesOffset = blockOffset;
        long quantsOffset = blockOffset + NVFP4_SUB_BLOCKS;
        long subQuantsOffset = quantsOffset;
        int outBase = dstOffset;
        int packedScales = segment.get(LE_INT, scalesOffset);
        for (int sub = 0; sub < NVFP4_SUB_BLOCKS; sub++) {
            float d = ue4m3ToF32(unsignedByte(packedScales, sub * 8));
            dequantizeNibble16(segment, subQuantsOffset, d, MXFP4_NIBBLE_PAIRS, dst, outBase);
            subQuantsOffset += NVFP4_SUB_BLOCK_SIZE / 2;
            outBase += NVFP4_SUB_BLOCK_SIZE;
        }
    }

    private static void dequantizeNibble16(
            MemorySegment segment,
            long quantsOffset,
            float scale,
            byte[] pairValues,
            float[] dst,
            int dstOffset) {
        long packed = segment.get(LE_LONG, quantsOffset);
        dequantizePackedNibbleByte8(packed, 0, scale, pairValues, dst, dstOffset, 0);
        dequantizePackedNibbleByte8(packed, 8, scale, pairValues, dst, dstOffset, 1);
        dequantizePackedNibbleByte8(packed, 16, scale, pairValues, dst, dstOffset, 2);
        dequantizePackedNibbleByte8(packed, 24, scale, pairValues, dst, dstOffset, 3);
        dequantizePackedNibbleByte8(packed, 32, scale, pairValues, dst, dstOffset, 4);
        dequantizePackedNibbleByte8(packed, 40, scale, pairValues, dst, dstOffset, 5);
        dequantizePackedNibbleByte8(packed, 48, scale, pairValues, dst, dstOffset, 6);
        dequantizePackedNibbleByte8(packed, 56, scale, pairValues, dst, dstOffset, 7);
    }

    private static void dequantizeNibble32(
            MemorySegment segment,
            long quantsOffset,
            float scale,
            byte[] pairValues,
            float[] dst,
            int dstOffset) {
        long first = segment.get(LE_LONG, quantsOffset);
        long second = segment.get(LE_LONG, quantsOffset + Long.BYTES);
        dequantizePackedNibbleByte16(first, 0, scale, pairValues, dst, dstOffset, 0);
        dequantizePackedNibbleByte16(first, 8, scale, pairValues, dst, dstOffset, 1);
        dequantizePackedNibbleByte16(first, 16, scale, pairValues, dst, dstOffset, 2);
        dequantizePackedNibbleByte16(first, 24, scale, pairValues, dst, dstOffset, 3);
        dequantizePackedNibbleByte16(first, 32, scale, pairValues, dst, dstOffset, 4);
        dequantizePackedNibbleByte16(first, 40, scale, pairValues, dst, dstOffset, 5);
        dequantizePackedNibbleByte16(first, 48, scale, pairValues, dst, dstOffset, 6);
        dequantizePackedNibbleByte16(first, 56, scale, pairValues, dst, dstOffset, 7);
        dequantizePackedNibbleByte16(second, 0, scale, pairValues, dst, dstOffset, 8);
        dequantizePackedNibbleByte16(second, 8, scale, pairValues, dst, dstOffset, 9);
        dequantizePackedNibbleByte16(second, 16, scale, pairValues, dst, dstOffset, 10);
        dequantizePackedNibbleByte16(second, 24, scale, pairValues, dst, dstOffset, 11);
        dequantizePackedNibbleByte16(second, 32, scale, pairValues, dst, dstOffset, 12);
        dequantizePackedNibbleByte16(second, 40, scale, pairValues, dst, dstOffset, 13);
        dequantizePackedNibbleByte16(second, 48, scale, pairValues, dst, dstOffset, 14);
        dequantizePackedNibbleByte16(second, 56, scale, pairValues, dst, dstOffset, 15);
    }

    private static void dequantizePackedNibbleByte8(
            long packed,
            int shift,
            float scale,
            byte[] pairValues,
            float[] dst,
            int dstOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        int pairBase = nibblePairBase(quant);
        dst[dstOffset + index] = scale * pairValues[pairBase];
        dst[dstOffset + 8 + index] = scale * pairValues[pairBase + 1];
    }

    private static void dequantizePackedNibbleByte16(
            long packed,
            int shift,
            float scale,
            byte[] pairValues,
            float[] dst,
            int dstOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        int pairBase = nibblePairBase(quant);
        dst[dstOffset + index] = scale * pairValues[pairBase];
        dst[dstOffset + 16 + index] = scale * pairValues[pairBase + 1];
    }

    private static void dequantizeQ4Block(
            MemorySegment segment,
            long quantsOffset,
            float scale,
            float bias,
            int zeroPoint,
            float[] dst,
            int dstOffset) {
        dequantizePackedQ4Long(segment.get(LE_LONG, quantsOffset), scale, bias, zeroPoint, dst, dstOffset, 0);
        dequantizePackedQ4Long(
                segment.get(LE_LONG, quantsOffset + Long.BYTES),
                scale,
                bias,
                zeroPoint,
                dst,
                dstOffset,
                8);
    }

    private static void dequantizePackedQ4Long(
            long packed,
            float scale,
            float bias,
            int zeroPoint,
            float[] dst,
            int dstOffset,
            int indexBase) {
        dequantizePackedQ4Byte(packed, 0, scale, bias, zeroPoint, dst, dstOffset, indexBase);
        dequantizePackedQ4Byte(packed, 8, scale, bias, zeroPoint, dst, dstOffset, indexBase + 1);
        dequantizePackedQ4Byte(packed, 16, scale, bias, zeroPoint, dst, dstOffset, indexBase + 2);
        dequantizePackedQ4Byte(packed, 24, scale, bias, zeroPoint, dst, dstOffset, indexBase + 3);
        dequantizePackedQ4Byte(packed, 32, scale, bias, zeroPoint, dst, dstOffset, indexBase + 4);
        dequantizePackedQ4Byte(packed, 40, scale, bias, zeroPoint, dst, dstOffset, indexBase + 5);
        dequantizePackedQ4Byte(packed, 48, scale, bias, zeroPoint, dst, dstOffset, indexBase + 6);
        dequantizePackedQ4Byte(packed, 56, scale, bias, zeroPoint, dst, dstOffset, indexBase + 7);
    }

    private static void dequantizePackedQ4Byte(
            long packed,
            int shift,
            float scale,
            float bias,
            int zeroPoint,
            float[] dst,
            int dstOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        dst[dstOffset + index] = ((quant & 0x0F) - zeroPoint) * scale + bias;
        dst[dstOffset + 16 + index] = ((quant >>> 4) - zeroPoint) * scale + bias;
    }

    private static void dequantizeQ5Block(
            MemorySegment segment,
            long quantsOffset,
            int highBits,
            float scale,
            float bias,
            int zeroPoint,
            float[] dst,
            int dstOffset) {
        dequantizePackedQ5Long(segment.get(LE_LONG, quantsOffset), highBits, scale, bias, zeroPoint, dst, dstOffset, 0);
        dequantizePackedQ5Long(
                segment.get(LE_LONG, quantsOffset + Long.BYTES),
                highBits,
                scale,
                bias,
                zeroPoint,
                dst,
                dstOffset,
                8);
    }

    private static void dequantizePackedQ5Long(
            long packed,
            int highBits,
            float scale,
            float bias,
            int zeroPoint,
            float[] dst,
            int dstOffset,
            int indexBase) {
        dequantizePackedQ5Byte(packed, 0, highBits, indexBase, scale, bias, zeroPoint, dst, dstOffset);
        dequantizePackedQ5Byte(packed, 8, highBits, indexBase + 1, scale, bias, zeroPoint, dst, dstOffset);
        dequantizePackedQ5Byte(packed, 16, highBits, indexBase + 2, scale, bias, zeroPoint, dst, dstOffset);
        dequantizePackedQ5Byte(packed, 24, highBits, indexBase + 3, scale, bias, zeroPoint, dst, dstOffset);
        dequantizePackedQ5Byte(packed, 32, highBits, indexBase + 4, scale, bias, zeroPoint, dst, dstOffset);
        dequantizePackedQ5Byte(packed, 40, highBits, indexBase + 5, scale, bias, zeroPoint, dst, dstOffset);
        dequantizePackedQ5Byte(packed, 48, highBits, indexBase + 6, scale, bias, zeroPoint, dst, dstOffset);
        dequantizePackedQ5Byte(packed, 56, highBits, indexBase + 7, scale, bias, zeroPoint, dst, dstOffset);
    }

    private static void dequantizePackedQ5Byte(
            long packed,
            int shift,
            int highBits,
            int index,
            float scale,
            float bias,
            int zeroPoint,
            float[] dst,
            int dstOffset) {
        int quant = unsignedByte(packed, shift);
        int low = (quant & 0x0F) | (((highBits >>> index) & 1) << 4);
        int high = (quant >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
        dst[dstOffset + index] = (low - zeroPoint) * scale + bias;
        dst[dstOffset + 16 + index] = (high - zeroPoint) * scale + bias;
    }

    private static void dequantizePackedQ1Byte(long masks, int shift, float scale, float[] dst, int dstOffset) {
        int mask = unsignedByte(masks, shift);
        int signBase = q1_0SignBase(mask);
        dst[dstOffset] = Q1_0_SIGNS[signBase] * scale;
        dst[dstOffset + 1] = Q1_0_SIGNS[signBase + 1] * scale;
        dst[dstOffset + 2] = Q1_0_SIGNS[signBase + 2] * scale;
        dst[dstOffset + 3] = Q1_0_SIGNS[signBase + 3] * scale;
        dst[dstOffset + 4] = Q1_0_SIGNS[signBase + 4] * scale;
        dst[dstOffset + 5] = Q1_0_SIGNS[signBase + 5] * scale;
        dst[dstOffset + 6] = Q1_0_SIGNS[signBase + 6] * scale;
        dst[dstOffset + 7] = Q1_0_SIGNS[signBase + 7] * scale;
    }

    private static void dequantizePackedQ1Long(long masks, float scale, float[] dst, int dstOffset) {
        dequantizePackedQ1Byte(masks, 0, scale, dst, dstOffset);
        dequantizePackedQ1Byte(masks, 8, scale, dst, dstOffset + 8);
        dequantizePackedQ1Byte(masks, 16, scale, dst, dstOffset + 16);
        dequantizePackedQ1Byte(masks, 24, scale, dst, dstOffset + 24);
        dequantizePackedQ1Byte(masks, 32, scale, dst, dstOffset + 32);
        dequantizePackedQ1Byte(masks, 40, scale, dst, dstOffset + 40);
        dequantizePackedQ1Byte(masks, 48, scale, dst, dstOffset + 48);
        dequantizePackedQ1Byte(masks, 56, scale, dst, dstOffset + 56);
    }

    static void dequantizeIQ4XSBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        int scalesH = segment.get(LE_SHORT, blockOffset + 2) & 0xFFFF;
        long scalesLOffset = blockOffset + 4;
        int scalesL = segment.get(LE_INT, scalesLOffset);
        long quantsOffset = blockOffset + 8;
        long groupQuantsOffset = quantsOffset;
        int outBase = dstOffset;
        for (int group = 0; group < IQ4_XS_GROUPS; group++) {
            float dl = d * (iq4XSScalePacked(scalesH, scalesL, group) - 32);
            dequantizeNibble32(segment, groupQuantsOffset, dl, IQ4_NL_NIBBLE_PAIRS, dst, outBase);
            groupQuantsOffset += IQ4_XS_GROUP_SIZE / 2;
            outBase += IQ4_XS_GROUP_SIZE;
        }
    }
}
