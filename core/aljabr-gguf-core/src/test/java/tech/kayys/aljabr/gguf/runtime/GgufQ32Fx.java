package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;

/**
 * Compatibility facade for Q32-family GGUF fixtures split by block family.
 */
final class GgufQ32Fx {
    private GgufQ32Fx() {
    }

    static void writeQ4_0Block(MemorySegment block, short scale, byte packedQuant) {
        GgufQ32Q4Fx.writeQ4_0Block(block, scale, packedQuant);
    }

    static void writeQ4_0LaneOrderBlock(MemorySegment block, short scale) {
        GgufQ32Q4Fx.writeQ4_0LaneOrderBlock(block, scale);
    }

    static void writeQ4_1Block(MemorySegment block, short scale, short min, byte packedQuant) {
        GgufQ32Q4Fx.writeQ4_1Block(block, scale, min, packedQuant);
    }

    static void writeQ4_1LaneOrderBlock(MemorySegment block, short scale, short min) {
        GgufQ32Q4Fx.writeQ4_1LaneOrderBlock(block, scale, min);
    }

    static float[] expectedQ4SmallLaneOrderRow(int zeroPoint, float bias) {
        return GgufQ32Q4Fx.expectedLaneOrderRow(zeroPoint, bias);
    }

    static byte[] expectedQ4SmallPreparedQuants(int zeroPoint) {
        return GgufQ32Q4Fx.expectedPreparedQuants(zeroPoint);
    }

    static void writeQ5_0Block(MemorySegment block, short scale, int highBits, byte packedQuant) {
        GgufQ32Q5Fx.writeQ5_0Block(block, scale, highBits, packedQuant);
    }

    static void writeQ5_0LaneOrderBlock(MemorySegment block, short scale, int highBits) {
        GgufQ32Q5Fx.writeQ5_0LaneOrderBlock(block, scale, highBits);
    }

    static void writeQ5_1Block(MemorySegment block, short scale, short min, int highBits, byte packedQuant) {
        GgufQ32Q5Fx.writeQ5_1Block(block, scale, min, highBits, packedQuant);
    }

    static void writeQ5_1LaneOrderBlock(MemorySegment block, short scale, short min, int highBits) {
        GgufQ32Q5Fx.writeQ5_1LaneOrderBlock(block, scale, min, highBits);
    }

    static float[] expectedQ5SmallLaneOrderRow(int highBits, int zeroPoint, float bias) {
        return GgufQ32Q5Fx.expectedLaneOrderRow(highBits, zeroPoint, bias);
    }

    static byte[] expectedQ5SmallPreparedQuants(int highBits, int zeroPoint) {
        return GgufQ32Q5Fx.expectedPreparedQuants(highBits, zeroPoint);
    }
}
