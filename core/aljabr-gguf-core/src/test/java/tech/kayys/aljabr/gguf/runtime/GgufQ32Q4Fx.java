package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Q32-family Q4_0 and Q4_1 block writers plus expected lane-order rows.
 */
final class GgufQ32Q4Fx {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ32Q4Fx() {
    }

    static void writeQ4_0Block(MemorySegment block, short scale, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        writePackedQuants(block, 2, packedQuant);
    }

    static void writeQ4_0LaneOrderBlock(MemorySegment block, short scale) {
        block.set(LE_SHORT, 0, scale);
        GgufNibFx.writeLaneOrder(block, 2);
    }

    static void writeQ4_1Block(MemorySegment block, short scale, short min, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, min);
        writePackedQuants(block, 4, packedQuant);
    }

    static void writeQ4_1LaneOrderBlock(MemorySegment block, short scale, short min) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, min);
        GgufNibFx.writeLaneOrder(block, 4);
    }

    static float[] expectedLaneOrderRow(int zeroPoint, float bias) {
        float[] expected = new float[32];
        for (int i = 0; i < 16; i++) {
            expected[i] = GgufNibFx.low(i) - zeroPoint + bias;
            expected[16 + i] = GgufNibFx.high(i) - zeroPoint + bias;
        }
        return expected;
    }

    static byte[] expectedPreparedQuants(int zeroPoint) {
        byte[] expected = new byte[32];
        for (int i = 0; i < 16; i++) {
            expected[i] = (byte) (GgufNibFx.low(i) - zeroPoint);
            expected[16 + i] = (byte) (GgufNibFx.high(i) - zeroPoint);
        }
        return expected;
    }

    private static void writePackedQuants(MemorySegment block, long offset, byte packedQuant) {
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, offset + i, packedQuant);
        }
    }
}
