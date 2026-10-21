package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Compatibility facade for K-quant fixture helpers split by family.
 */
final class GgufKFx {
    static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufKFx() {
    }

    static void writeQ2KBlock(MemorySegment block, byte scale, byte packedQuant) {
        GgufQ2Fx.writeBlock(block, scale, packedQuant);
    }

    static void writeQ2KNoMinLaneOrderBlock(MemorySegment block) {
        GgufQ2Fx.writeNoMinLaneOrderBlock(block);
    }

    static void writeQ2KMinLaneOrderBlock(MemorySegment block) {
        GgufQ2Fx.writeMinLaneOrderBlock(block);
    }

    static void writeQ2KBlockWithMin(MemorySegment block, byte scaleMin, byte packedQuant) {
        GgufQ2Fx.writeBlockWithMin(block, scaleMin, packedQuant);
    }

    static float expectedQ2LaneOrderDot(float[] vector, boolean hasMins) {
        return GgufQ2Fx.expectedLaneOrderDot(vector, hasMins);
    }

    static float[] expectedQ2LaneOrderRow(boolean hasMins) {
        return GgufQ2Fx.expectedLaneOrderRow(hasMins);
    }

    static void writeQ3KBlock(MemorySegment block, int signedScale, byte packedQuant, byte highMask) {
        GgufQ3Fx.writeBlock(block, signedScale, packedQuant, highMask);
    }

    static void writeQ3KLaneOrderBlock(MemorySegment block) {
        GgufQ3LaneFx.writeBlock(block);
    }

    static float expectedQ3LaneOrderDot(float[] vector) {
        return GgufQ3LaneFx.expectedDot(vector);
    }

    static float[] expectedQ3LaneOrderRow() {
        return GgufQ3LaneFx.expectedRow();
    }

    static void writeSimpleQ4KBlock(MemorySegment block) {
        GgufQ4Fx.writeSimpleBlock(block);
    }

    static void writeQ4KBlockWithAllScalesAndMins(MemorySegment block) {
        GgufQ4Fx.writeBlockWithAllScalesAndMins(block);
    }

    static void writeQ4KNoMinLaneOrderBlock(MemorySegment block) {
        GgufQ4Fx.writeNoMinLaneOrderBlock(block);
    }

    static void writeQ4KMinLaneOrderBlock(MemorySegment block) {
        GgufQ4Fx.writeMinLaneOrderBlock(block);
    }

    static float[] expectedQ4KLaneOrderRow(boolean hasMins) {
        return GgufQ4Fx.expectedLaneOrderRow(hasMins);
    }

    static float expectedQ4KLaneOrderDot(float[] vector, boolean hasMins) {
        return GgufQ4CalcFx.expectedDot(vector, hasMins);
    }

    static void writeQ5KBlock(MemorySegment block, byte highBits, byte packedQuant) {
        GgufQ5Fx.writeBlock(block, highBits, packedQuant);
    }

    static void writeQ5KBlock(MemorySegment block) {
        GgufQ5Fx.writeBlock(block);
    }

    static void writeQ5KNoMinLaneOrderBlock(MemorySegment block) {
        GgufQ5LaneFx.writeNoMinBlock(block);
    }

    static void writeQ5KMinLaneOrderBlock(MemorySegment block) {
        GgufQ5LaneFx.writeMinBlock(block);
    }

    static void writeQ5KBlockWithMin(
            MemorySegment block,
            byte scale,
            byte min,
            byte highBits,
            byte packedQuant) {
        GgufQ5Fx.writeBlockWithMin(block, scale, min, highBits, packedQuant);
    }

    static float expectedQ5LaneOrderDot(float[] vector, boolean hasMins) {
        return GgufQ5LaneFx.expectedDot(vector, hasMins);
    }

    static float[] expectedQ5LaneOrderRow(boolean hasMins) {
        return GgufQ5LaneFx.expectedRow(hasMins);
    }

    static void writeQ6KBlock(MemorySegment block, byte lowPacked, byte highPacked, byte scale) {
        GgufQ6Fx.writeBlock(block, lowPacked, highPacked, scale);
    }

    static void writeQ6KBlock(MemorySegment block) {
        GgufQ6Fx.writeBlock(block);
    }

    static void writeQ6KLaneOrderBlock(MemorySegment block) {
        GgufQ6Fx.writeLaneOrderBlock(block);
    }

    static int q6LaneOrderLowBits(int index) {
        return GgufQ6Fx.laneOrderLowBits(index);
    }

    static int q6LaneOrderHighBits(int index) {
        return GgufQ6Fx.laneOrderHighBits(index);
    }
}
