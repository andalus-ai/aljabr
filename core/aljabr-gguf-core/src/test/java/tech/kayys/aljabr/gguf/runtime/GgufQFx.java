package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
/**
 * Compatibility facade for compact-quant fixtures split by format family.
 */
final class GgufQFx {
    private GgufQFx() {
    }

    static void writeQ1_0Block(MemorySegment block, short scale, byte packedBits) {
        GgufLowFx.writeQ1_0Block(block, scale, packedBits);
    }

    static void writeQ1_0Block(MemorySegment block, byte packedBits) {
        GgufLowFx.writeQ1_0Block(block, packedBits);
    }

    static void writeTQ1_0Block(MemorySegment block, short scale, byte packedQuant) {
        GgufLowFx.writeTQ1_0Block(block, scale, packedQuant);
    }

    static void writeTQ1_0Block(MemorySegment block, byte packedQuant) {
        GgufLowFx.writeTQ1_0Block(block, packedQuant);
    }

    static void writeTQ2_0Block(MemorySegment block, short scale, byte packedQuant) {
        GgufLowFx.writeTQ2_0Block(block, scale, packedQuant);
    }

    static void writeTQ2_0Block(MemorySegment block, byte packedQuant) {
        GgufLowFx.writeTQ2_0Block(block, packedQuant);
    }

    static void writeMXFP4Block(MemorySegment block, byte exponent, byte packedQuant) {
        GgufF4Fx.writeMXFP4Block(block, exponent, packedQuant);
    }

    static void writeMXFP4Block(MemorySegment block, byte packedQuant) {
        GgufF4Fx.writeMXFP4Block(block, packedQuant);
    }

    static void writeNVFP4Block(MemorySegment block, byte scale, byte packedQuant) {
        GgufF4Fx.writeNVFP4Block(block, scale, packedQuant);
    }

    static void writeNVFP4Block(
            MemorySegment block,
            byte firstScale,
            byte secondScale,
            byte thirdScale,
            byte fourthScale,
            byte packedQuant) {
        GgufF4Fx.writeNVFP4Block(block, firstScale, secondScale, thirdScale, fourthScale, packedQuant);
    }

    static void writeNVFP4Block(MemorySegment block, byte packedQuant) {
        GgufF4Fx.writeNVFP4Block(block, packedQuant);
    }

    static void writeQ8Block(MemorySegment block, short scale, byte quant) {
        GgufQ8Fx.writeQ8Block(block, scale, quant);
    }

    static void writeQ8RampBlock(MemorySegment block, short scale, int start) {
        GgufQ8Fx.writeQ8RampBlock(block, scale, start);
    }

    static void writeQ8_1Block(MemorySegment block, short scale, byte quant) {
        GgufQ8Fx.writeQ8_1Block(block, scale, quant);
    }

    static void writeQ8_1Block(MemorySegment block, byte quant) {
        GgufQ8Fx.writeQ8_1Block(block, quant);
    }

    static void writeQ8KBlock(MemorySegment block, float scale, byte quant) {
        GgufQ8Fx.writeQ8KBlock(block, scale, quant);
    }

    static void writeIQ4NLBlock(MemorySegment block, short scale, byte packedQuant) {
        GgufIQFx.writeIQ4NLBlock(block, scale, packedQuant);
    }

    static void writeIQ4XSBlock(MemorySegment block, short scale, byte packedQuant) {
        GgufIQFx.writeIQ4XSBlock(block, scale, packedQuant);
    }

    static void writeIQ4XSBlock(MemorySegment block, byte packedQuant) {
        GgufIQFx.writeIQ4XSBlock(block, packedQuant);
    }
}
