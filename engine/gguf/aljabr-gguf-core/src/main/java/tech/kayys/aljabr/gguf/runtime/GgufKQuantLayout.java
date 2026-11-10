package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * K-quant scale/min unpacking helpers.
 *
 * <p>The K-family formats pack group scales, mins, and high bits differently;
 * this class isolates those layouts from the arithmetic kernels.</p>
 * @author bhangun
 */
final class GgufKQuantLayout {
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufKQuantLayout() {
    }

    static int iq4XSScalePacked(int scalesH, int scalesL, int group) {
        int lowByte = unsignedByte(scalesL, (group >>> 1) * 8);
        int low = (lowByte >>> (4 * (group & 1))) & 0x0F;
        int high = (scalesH >>> (2 * group)) & 0x03;
        return low | (high << 4);
    }

    static int iq4XSScale(int scalesH, byte[] data, int scalesLOffset, int group) {
        int low = ((data[scalesLOffset + group / 2] & 0xFF) >>> (4 * (group % 2))) & 0x0F;
        int high = (scalesH >>> (2 * group)) & 0x03;
        return low | (high << 4);
    }

    static int scaleMinK4PackedCode(long scalesLow, int scalesHigh, int index) {
        int scale;
        int min;
        if (index < 4) {
            scale = unsignedByte(scalesLow, index * 8) & 63;
            min = unsignedByte(scalesLow, (index + 4) * 8) & 63;
        } else {
            int high = unsignedByte(scalesHigh, (index - 4) * 8);
            scale = (high & 0x0F) | ((unsignedByte(scalesLow, (index - 4) * 8) >>> 6) << 4);
            min = (high >>> 4) | ((unsignedByte(scalesLow, index * 8) >>> 6) << 4);
        }
        return scale | (min << 8);
    }

    static int scaleFromPackedScaleMin(int packed) {
        return packed & 0xFF;
    }

    static int minFromPackedScaleMin(int packed) {
        return packed >>> 8;
    }

    static int scaleK4Packed(long scalesLow, int scalesHigh, int index) {
        if (index < 4) {
            return unsignedByte(scalesLow, index * 8) & 63;
        }
        return (unsignedByte(scalesHigh, (index - 4) * 8) & 0x0F)
                | ((unsignedByte(scalesLow, (index - 4) * 8) >>> 6) << 4);
    }

    static float minContribution(float dMin, int encodedMin) {
        return encodedMin == 0 ? 0.0f : dMin * encodedMin;
    }

    static int q3KHighBias(int highBits, int highMask) {
        return (((highBits & highMask) - 1) >>> 31) << 2;
    }

    static void unpackQ3KScales(MemorySegment segment, long scalesOffset, int[] scales) {
        long lowNibbles = segment.get(LE_LONG, scalesOffset);
        int highBitsPacked = segment.get(LE_INT, scalesOffset + Long.BYTES);
        for (int group = 0; group < 8; group++) {
            int lowByte = unsignedByte(lowNibbles, group * 8);
            int highByte = unsignedByte(highBitsPacked, (group & 3) * 8);
            int highBits = (highByte >>> (2 * (group >>> 2))) & 0x03;
            scales[group] = ((lowByte & 0x0F) | (highBits << 4)) - 32;
        }
        for (int group = 8; group < 16; group++) {
            int lowByte = unsignedByte(lowNibbles, (group - 8) * 8);
            int highByte = unsignedByte(highBitsPacked, (group & 3) * 8);
            int highBits = (highByte >>> (2 * (group >>> 2))) & 0x03;
            int lowBits = (lowByte >>> 4) & 0x0F;
            scales[group] = (lowBits | (highBits << 4)) - 32;
        }
    }

    static void unpackQ3KScales(byte[] data, int scalesOffset, int[] scales) {
        for (int group = 0; group < 16; group++) {
            int lowBits = group < 8
                    ? data[scalesOffset + group] & 0x0F
                    : (data[scalesOffset + group - 8] >>> 4) & 0x0F;
            int highBits = ((data[scalesOffset + 8 + (group % 4)] & 0xFF) >>> (2 * (group / 4))) & 0x03;
            scales[group] = (lowBits | (highBits << 4)) - 32;
        }
    }
}
