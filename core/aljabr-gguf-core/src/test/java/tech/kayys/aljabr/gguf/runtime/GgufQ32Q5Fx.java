package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Q32-family Q5_0 and Q5_1 block writers plus high-bit lane expectations.
 */
final class GgufQ32Q5Fx {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ32Q5Fx() {
    }

    static void writeQ5_0Block(MemorySegment block, short scale, int highBits, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_INT, 2, highBits);
        writePackedQuants(block, 6, packedQuant);
    }

    static void writeQ5_0LaneOrderBlock(MemorySegment block, short scale, int highBits) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_INT, 2, highBits);
        GgufNibFx.writeLaneOrder(block, 6);
    }

    static void writeQ5_1Block(MemorySegment block, short scale, short min, int highBits, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, min);
        block.set(LE_INT, 4, highBits);
        writePackedQuants(block, 8, packedQuant);
    }

    static void writeQ5_1LaneOrderBlock(MemorySegment block, short scale, short min, int highBits) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, min);
        block.set(LE_INT, 4, highBits);
        GgufNibFx.writeLaneOrder(block, 8);
    }

    static float[] expectedLaneOrderRow(int highBits, int zeroPoint, float bias) {
        float[] expected = new float[32];
        for (int i = 0; i < 16; i++) {
            expected[i] = lowValue(highBits, i) - zeroPoint + bias;
            expected[16 + i] = highValue(highBits, i) - zeroPoint + bias;
        }
        return expected;
    }

    static byte[] expectedPreparedQuants(int highBits, int zeroPoint) {
        byte[] expected = new byte[32];
        for (int i = 0; i < 16; i++) {
            expected[i] = (byte) (lowValue(highBits, i) - zeroPoint);
            expected[16 + i] = (byte) (highValue(highBits, i) - zeroPoint);
        }
        return expected;
    }

    private static void writePackedQuants(MemorySegment block, long offset, byte packedQuant) {
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, offset + i, packedQuant);
        }
    }

    private static int lowValue(int highBits, int index) {
        return GgufNibFx.low(index) | (((highBits >>> index) & 1) << 4);
    }

    private static int highValue(int highBits, int index) {
        return GgufNibFx.high(index) | (((highBits >>> (index + 16)) & 1) << 4);
    }
}
