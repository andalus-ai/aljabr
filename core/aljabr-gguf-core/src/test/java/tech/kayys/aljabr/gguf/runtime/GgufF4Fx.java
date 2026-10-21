package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * FP4-family compact block writers for GGUF runtime tests.
 */
final class GgufF4Fx {
    private GgufF4Fx() {
    }

    static void writeMXFP4Block(MemorySegment block, byte exponent, byte packedQuant) {
        block.set(ValueLayout.JAVA_BYTE, 0, exponent);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 1 + i, packedQuant);
        }
    }

    static void writeMXFP4Block(MemorySegment block, byte packedQuant) {
        writeMXFP4Block(block, (byte) 128, packedQuant);
    }

    static void writeNVFP4Block(MemorySegment block, byte scale, byte packedQuant) {
        writeNVFP4Block(block, scale, scale, scale, scale, packedQuant);
    }

    static void writeNVFP4Block(
            MemorySegment block,
            byte firstScale,
            byte secondScale,
            byte thirdScale,
            byte fourthScale,
            byte packedQuant) {
        block.set(ValueLayout.JAVA_BYTE, 0, firstScale);
        block.set(ValueLayout.JAVA_BYTE, 1, secondScale);
        block.set(ValueLayout.JAVA_BYTE, 2, thirdScale);
        block.set(ValueLayout.JAVA_BYTE, 3, fourthScale);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, packedQuant);
        }
    }

    static void writeNVFP4Block(MemorySegment block, byte packedQuant) {
        writeNVFP4Block(block, (byte) 0x40, packedQuant);
    }
}
