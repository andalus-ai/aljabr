package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q5_K lane-order fixtures covering packed nibbles, high-bit masks, and optional min terms.
 */
final class GgufQ5LaneFx {
    private GgufQ5LaneFx() {
    }

    static void writeNoMinBlock(MemorySegment block) {
        GgufQ5Fx.writeBlock(block, (byte) 0, (byte) 0);
        writeHighBits(block);
        writeQuants(block);
    }

    static void writeMinBlock(MemorySegment block) {
        GgufQ5Fx.writeBlockWithMin(block, (byte) 2, (byte) 1, (byte) 0, (byte) 0);
        writeHighBits(block);
        writeQuants(block);
    }

    static float expectedDot(float[] vector, boolean hasMins) {
        return GgufQ5CalcFx.expectedDot(vector, hasMins);
    }

    static float[] expectedRow(boolean hasMins) {
        return GgufQ5CalcFx.expectedRow(hasMins);
    }

    private static void writeHighBits(MemorySegment block) {
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) GgufQ5CalcFx.highBits(i));
        }
    }

    private static void writeQuants(MemorySegment block) {
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 48 + i, (byte) GgufQ5CalcFx.packedQuant(i));
        }
    }
}
