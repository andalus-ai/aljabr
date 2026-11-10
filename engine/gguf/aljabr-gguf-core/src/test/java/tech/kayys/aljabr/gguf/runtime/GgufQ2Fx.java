package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q2_K synthetic block builders for GGUF runtime tests.
 */
final class GgufQ2Fx {
    private GgufQ2Fx() {
    }

    static void writeBlock(MemorySegment block, byte scale, byte packedQuant) {
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, scale);
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, packedQuant);
        }
        block.set(GgufKFx.LE_SHORT, 80, (short) 0x3c00);
        block.set(GgufKFx.LE_SHORT, 82, (short) 0);
    }

    static void writeNoMinLaneOrderBlock(MemorySegment block) {
        writeBlock(block, (byte) 1, (byte) 0);
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) GgufQ2LaneFx.quant(i));
        }
    }

    static void writeMinLaneOrderBlock(MemorySegment block) {
        writeBlockWithMin(block, (byte) 0x12, (byte) 0);
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) GgufQ2LaneFx.quant(i));
        }
    }

    static void writeBlockWithMin(MemorySegment block, byte scaleMin, byte packedQuant) {
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, scaleMin);
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, packedQuant);
        }
        block.set(GgufKFx.LE_SHORT, 80, (short) 0x3c00);
        block.set(GgufKFx.LE_SHORT, 82, (short) 0x3c00);
    }

    static float expectedLaneOrderDot(float[] vector, boolean hasMins) {
        return GgufQ2LaneFx.expectedDot(vector, hasMins);
    }

    static float[] expectedLaneOrderRow(boolean hasMins) {
        return GgufQ2LaneFx.expectedRow(hasMins);
    }
}
