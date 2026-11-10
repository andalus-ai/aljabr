package tech.kayys.aljabr.gguf.runtime;

/**
 * Q6_K lane-order expectations for packed low nibbles and two-bit high lanes.
 */
final class GgufQ6LaneFx {
    private GgufQ6LaneFx() {
    }

    static float expectedDot(float[] vector) {
        float expected = 0.0f;
        for (int superBlock = 0; superBlock < 256; superBlock += 128) {
            int lowBase = (superBlock / 128) * 64;
            int highBase = (superBlock / 128) * 32;
            for (int i = 0; i < 32; i++) {
                int lowA = GgufKFx.q6LaneOrderLowBits(lowBase + i);
                int lowB = GgufKFx.q6LaneOrderLowBits(lowBase + 32 + i);
                int highBits = GgufKFx.q6LaneOrderHighBits(highBase + i);
                expected += lowValue(lowA, highBits) * vector[superBlock + i];
                expected += highLowValue(lowB, highBits) * vector[superBlock + 32 + i];
                expected += highNibbleValue(lowA, highBits) * vector[superBlock + 64 + i];
                expected += highHighValue(lowB, highBits) * vector[superBlock + 96 + i];
            }
        }
        return expected;
    }

    static float[] expectedRow() {
        float[] expected = new float[256];
        for (int superBlock = 0; superBlock < 256; superBlock += 128) {
            int lowBase = (superBlock / 128) * 64;
            int highBase = (superBlock / 128) * 32;
            for (int i = 0; i < 32; i++) {
                int lowA = GgufKFx.q6LaneOrderLowBits(lowBase + i);
                int lowB = GgufKFx.q6LaneOrderLowBits(lowBase + 32 + i);
                int highBits = GgufKFx.q6LaneOrderHighBits(highBase + i);
                expected[superBlock + i] = lowValue(lowA, highBits);
                expected[superBlock + 32 + i] = highLowValue(lowB, highBits);
                expected[superBlock + 64 + i] = highNibbleValue(lowA, highBits);
                expected[superBlock + 96 + i] = highHighValue(lowB, highBits);
            }
        }
        return expected;
    }

    private static int lowValue(int lowPacked, int highBits) {
        return ((lowPacked & 0x0F) | ((highBits & 0x03) << 4)) - 32;
    }

    private static int highLowValue(int lowPacked, int highBits) {
        return ((lowPacked & 0x0F) | (((highBits >>> 2) & 0x03) << 4)) - 32;
    }

    private static int highNibbleValue(int lowPacked, int highBits) {
        return ((lowPacked >>> 4) | (((highBits >>> 4) & 0x03) << 4)) - 32;
    }

    private static int highHighValue(int lowPacked, int highBits) {
        return ((lowPacked >>> 4) | (((highBits >>> 6) & 0x03) << 4)) - 32;
    }
}
