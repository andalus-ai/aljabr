package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q3_K synthetic block builders.
 */
final class GgufQ3Fx {
    private GgufQ3Fx() {
    }

    static void writeBlock(MemorySegment block, int signedScale, byte packedQuant, byte highMask) {
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, highMask);
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 32 + i, packedQuant);
        }
        writeScales(block, signedScale);
        block.set(GgufKFx.LE_SHORT, 108, (short) 0x3c00);
    }

    private static void writeScales(MemorySegment block, int signedScale) {
        int encoded = Math.max(0, Math.min(63, signedScale + 32));
        for (int group = 0; group < 16; group++) {
            int low = encoded & 0x0F;
            int high = (encoded >>> 4) & 0x03;
            long lowOffset = 96L + (group < 8 ? group : group - 8);
            int lowByte = block.get(ValueLayout.JAVA_BYTE, lowOffset) & 0xFF;
            if (group < 8) {
                lowByte = (lowByte & 0xF0) | low;
            } else {
                lowByte = (lowByte & 0x0F) | (low << 4);
            }
            block.set(ValueLayout.JAVA_BYTE, lowOffset, (byte) lowByte);

            long highOffset = 96L + 8 + (group % 4);
            int highByte = block.get(ValueLayout.JAVA_BYTE, highOffset) & 0xFF;
            highByte |= high << (2 * (group / 4));
            block.set(ValueLayout.JAVA_BYTE, highOffset, (byte) highByte);
        }
    }
}
