package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Byte-level read helpers shared by GGUF tensor kernels.
 *
 * <p>Centralizing little-endian and packed-byte decoding keeps quant format
 * code consistent across heap arrays and foreign memory segments.</p>
 * @author bhangun
 */
final class GgufByteValues {
    private GgufByteValues() {
    }

    static int u8(MemorySegment segment, long offset) {
        return segment.get(ValueLayout.JAVA_BYTE, offset) & 0xFF;
    }

    static int u8(byte[] data, long offset) {
        return data[(int) offset] & 0xFF;
    }

    static int signedByte(long packed, int shift) {
        return (byte) (packed >>> shift);
    }

    static int unsignedByte(long packed, int shift) {
        return (int) (packed >>> shift) & 0xFF;
    }

    static int unsignedByte(int packed, int shift) {
        return (packed >>> shift) & 0xFF;
    }

    static short leShort(byte[] data, long offset) {
        int index = (int) offset;
        return (short) ((data[index] & 0xFF) | ((data[index + 1] & 0xFF) << 8));
    }

    static int leInt(byte[] data, long offset) {
        int index = (int) offset;
        return (data[index] & 0xFF)
                | ((data[index + 1] & 0xFF) << 8)
                | ((data[index + 2] & 0xFF) << 16)
                | ((data[index + 3] & 0xFF) << 24);
    }

    static float leFloat(byte[] data, long offset) {
        return Float.intBitsToFloat(leInt(data, offset));
    }
}
