package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * IQ compact block writers for GGUF runtime tests.
 */
final class GgufIQFx {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufIQFx() {
    }

    static void writeIQ4NLBlock(MemorySegment block, short scale, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, packedQuant);
        }
    }

    static void writeIQ4XSBlock(MemorySegment block, short scale, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, (short) 0xAAAA);
        for (int i = 0; i < 4; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, (byte) 0x11);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 8 + i, packedQuant);
        }
    }

    static void writeIQ4XSBlock(MemorySegment block, byte packedQuant) {
        writeIQ4XSBlock(block, (short) 0x3c00, packedQuant);
    }
}
