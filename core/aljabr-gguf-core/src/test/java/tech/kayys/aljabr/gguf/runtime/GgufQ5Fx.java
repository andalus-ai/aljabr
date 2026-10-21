package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q5_K synthetic block builders for high-bit and min-code layouts.
 */
final class GgufQ5Fx {
    private GgufQ5Fx() {
    }

    static void writeBlock(MemorySegment block, byte highBits, byte packedQuant) {
        block.set(GgufKFx.LE_SHORT, 0, (short) 0x3c00);
        block.set(GgufKFx.LE_SHORT, 2, (short) 0);

        byte[] scales = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
        for (int i = 0; i < scales.length; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, scales[i]);
        }
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, highBits);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 48 + i, packedQuant);
        }
    }

    static void writeBlock(MemorySegment block) {
        writeBlock(block, (byte) 0xFF, (byte) 0);
    }

    static void writeBlockWithMin(
            MemorySegment block,
            byte scale,
            byte min,
            byte highBits,
            byte packedQuant) {
        block.set(GgufKFx.LE_SHORT, 0, (short) 0x3c00);
        block.set(GgufKFx.LE_SHORT, 2, (short) 0x3c00);

        int scaleNibble = scale & 0x0F;
        int minNibble = min & 0x0F;
        for (int i = 0; i < 4; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, (byte) scaleNibble);
            block.set(ValueLayout.JAVA_BYTE, 8 + i, (byte) minNibble);
            block.set(ValueLayout.JAVA_BYTE, 12 + i, (byte) ((minNibble << 4) | scaleNibble));
        }
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, highBits);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 48 + i, packedQuant);
        }
    }
}
