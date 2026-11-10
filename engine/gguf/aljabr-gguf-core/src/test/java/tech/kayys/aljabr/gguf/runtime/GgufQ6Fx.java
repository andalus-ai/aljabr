package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q6_K synthetic block builders and lane-order generators.
 */
final class GgufQ6Fx {
    private GgufQ6Fx() {
    }

    static void writeBlock(MemorySegment block, byte lowPacked, byte highPacked, byte scale) {
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, lowPacked);
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 128 + i, highPacked);
        }
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 192 + i, scale);
        }
        block.set(GgufKFx.LE_SHORT, 208, (short) 0x3c00);
    }

    static void writeBlock(MemorySegment block) {
        writeBlock(block, (byte) 0x11, (byte) 0xAA, (byte) 1);
    }

    static void writeLaneOrderBlock(MemorySegment block) {
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, (byte) laneOrderLowBits(i));
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 128 + i, (byte) laneOrderHighBits(i));
        }
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 192 + i, (byte) 1);
        }
        block.set(GgufKFx.LE_SHORT, 208, (short) 0x3c00);
    }

    static int laneOrderLowBits(int index) {
        int low = index & 0x0F;
        int high = 15 - low;
        return low | (high << 4);
    }

    static int laneOrderHighBits(int index) {
        return (index * 53 + 0xA7) & 0xFF;
    }
}
