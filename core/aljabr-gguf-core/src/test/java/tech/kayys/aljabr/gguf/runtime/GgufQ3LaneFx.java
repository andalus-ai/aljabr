package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q3_K lane-order fixtures that verify packed 2-bit values and high-mask lanes independently.
 */
final class GgufQ3LaneFx {
    private GgufQ3LaneFx() {
    }

    static void writeBlock(MemorySegment block) {
        GgufQ3Fx.writeBlock(block, 1, (byte) 0, (byte) 0);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, (byte) GgufQ3CalcFx.highBits(i));
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 32 + i, (byte) GgufQ3CalcFx.quant(i));
        }
    }

    static float expectedDot(float[] vector) {
        return GgufQ3CalcFx.expectedDot(vector);
    }

    static float[] expectedRow() {
        return GgufQ3CalcFx.expectedRow();
    }
}
