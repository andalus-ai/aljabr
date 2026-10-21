package tech.kayys.aljabr.gguf.runtime;

/**
 * Q5_K lane-order expectation math shared by high-bit and min-code tests.
 */
final class GgufQ5CalcFx {
    private GgufQ5CalcFx() {
    }

    static int highBits(int index) {
        return (index * 73 + 0x5A) & 0xFF;
    }

    static int packedQuant(int index) {
        int low = index & 0x0F;
        int high = 15 - low;
        return low | (high << 4);
    }

    static float expectedDot(float[] vector, boolean hasMins) {
        float expected = 0.0f;
        for (int superBlock = 0; superBlock < 256; superBlock += 64) {
            expected += superBlockDot(vector, superBlock, hasMins);
        }
        return expected;
    }

    static float[] expectedRow(boolean hasMins) {
        float[] expected = new float[256];
        for (int superBlock = 0; superBlock < 256; superBlock += 64) {
            fillSuperBlockRow(expected, superBlock, hasMins);
        }
        return expected;
    }

    private static float superBlockDot(float[] vector, int superBlock, boolean hasMins) {
        float expected = 0.0f;
        int quantBase = (superBlock / 64) * 32;
        int highMaskLow = 1 << (superBlock / 32);
        int highMaskHigh = highMaskLow << 1;
        int highShiftLow = superBlock / 32;
        int highShiftHigh = highShiftLow + 1;
        for (int i = 0; i < 32; i++) {
            int highBits = highBits(i);
            expected += value(lowValue(quantBase, i, highBits, highMaskLow, highShiftLow), hasMins)
                    * vector[superBlock + i];
            expected += value(highValue(quantBase, i, highBits, highMaskHigh, highShiftHigh), hasMins)
                    * vector[superBlock + 32 + i];
        }
        return expected;
    }

    private static void fillSuperBlockRow(float[] expected, int superBlock, boolean hasMins) {
        int quantBase = (superBlock / 64) * 32;
        int highMaskLow = 1 << (superBlock / 32);
        int highMaskHigh = highMaskLow << 1;
        int highShiftLow = superBlock / 32;
        int highShiftHigh = highShiftLow + 1;
        for (int i = 0; i < 32; i++) {
            int highBits = highBits(i);
            expected[superBlock + i] = value(lowValue(quantBase, i, highBits, highMaskLow, highShiftLow), hasMins);
            expected[superBlock + 32 + i] =
                    value(highValue(quantBase, i, highBits, highMaskHigh, highShiftHigh), hasMins);
        }
    }

    private static int lowValue(int quantBase, int index, int highBits, int highMask, int highShift) {
        int low = (quantBase + index) & 0x0F;
        return low + (((highBits & highMask) >>> highShift) << 4);
    }

    private static int highValue(int quantBase, int index, int highBits, int highMask, int highShift) {
        int high = 15 - ((quantBase + index) & 0x0F);
        return high + (((highBits & highMask) >>> highShift) << 4);
    }

    private static float value(int value, boolean hasMins) {
        return hasMins ? 2.0f * value - 1.0f : value;
    }
}
