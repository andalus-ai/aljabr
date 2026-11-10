package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q4_K synthetic block builders for min and no-min layouts.
 */
final class GgufQ4Fx {
    private GgufQ4Fx() {
    }

    static void writeSimpleBlock(MemorySegment block) {
        block.set(GgufKFx.LE_SHORT, 0, (short) 0x3c00);
        block.set(GgufKFx.LE_SHORT, 2, (short) 0);

        byte[] scales = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
        for (int i = 0; i < scales.length; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, scales[i]);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) 0x21);
        }
    }

    static void writeBlockWithAllScalesAndMins(MemorySegment block) {
        block.set(GgufKFx.LE_SHORT, 0, (short) 0x3c00);
        block.set(GgufKFx.LE_SHORT, 2, (short) 0x3800);

        byte[] scales = {1, 1, 1, 1, 2, 2, 2, 2, 0x21, 0x21, 0x21, 0x21};
        for (int i = 0; i < scales.length; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, scales[i]);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) 0x21);
        }
    }

    static void writeNoMinLaneOrderBlock(MemorySegment block) {
        writeSimpleBlock(block);
        writeLaneOrderQuants(block);
    }

    static void writeMinLaneOrderBlock(MemorySegment block) {
        writeBlockWithAllScalesAndMins(block);
        writeLaneOrderQuants(block);
    }

    static float[] expectedLaneOrderRow(boolean hasMins) {
        return GgufQ4CalcFx.expectedRow(hasMins);
    }

    private static void writeLaneOrderQuants(MemorySegment block) {
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) GgufQ4CalcFx.packedQuant(i));
        }
    }
}
