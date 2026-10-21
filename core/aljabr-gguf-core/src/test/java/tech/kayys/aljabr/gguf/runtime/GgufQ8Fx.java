package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Q8-family compact block writers for GGUF runtime tests.
 */
final class GgufQ8Fx {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ8Fx() {
    }

    static void writeQ8Block(MemorySegment block, short scale, byte quant) {
        block.set(LE_SHORT, 0, scale);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, quant);
        }
    }

    static void writeQ8RampBlock(MemorySegment block, short scale, int start) {
        block.set(LE_SHORT, 0, scale);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, (byte) (start < 0 ? start + i : start - i));
        }
    }

    static void writeQ8_1Block(MemorySegment block, short scale, byte quant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, (short) 0x7b00);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, quant);
        }
    }

    static void writeQ8_1Block(MemorySegment block, byte quant) {
        writeQ8_1Block(block, (short) 0x3c00, quant);
    }

    static void writeQ8KBlock(MemorySegment block, float scale, byte quant) {
        block.set(LE_FLOAT, 0, scale);
        for (int i = 0; i < 256; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, quant);
        }
        short groupSum = (short) (quant * 16);
        for (int group = 0; group < 16; group++) {
            block.set(LE_SHORT, 260 + group * 2L, groupSum);
        }
    }
}
