package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Low-bit and ternary compact-quant block writers for GGUF runtime tests.
 */
final class GgufLowFx {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufLowFx() {
    }

    static void writeQ1_0Block(MemorySegment block, short scale, byte packedBits) {
        block.set(LE_SHORT, 0, scale);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, packedBits);
        }
    }

    static void writeQ1_0Block(MemorySegment block, byte packedBits) {
        writeQ1_0Block(block, (short) 0x3c00, packedBits);
    }

    static void writeTQ1_0Block(MemorySegment block, short scale, byte packedQuant) {
        for (int i = 0; i < 48; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, packedQuant);
        }
        for (int i = 0; i < 4; i++) {
            block.set(ValueLayout.JAVA_BYTE, 48 + i, packedQuant);
        }
        block.set(LE_SHORT, 52, scale);
    }

    static void writeTQ1_0Block(MemorySegment block, byte packedQuant) {
        writeTQ1_0Block(block, (short) 0x3c00, packedQuant);
    }

    static void writeTQ2_0Block(MemorySegment block, short scale, byte packedQuant) {
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, packedQuant);
        }
        block.set(LE_SHORT, 64, scale);
    }

    static void writeTQ2_0Block(MemorySegment block, byte packedQuant) {
        writeTQ2_0Block(block, (short) 0x3c00, packedQuant);
    }
}
