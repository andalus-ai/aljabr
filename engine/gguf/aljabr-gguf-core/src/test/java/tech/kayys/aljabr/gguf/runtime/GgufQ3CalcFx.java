package tech.kayys.aljabr.gguf.runtime;

/**
 * Q3_K lane-order expectation math shared by packed quant and high-mask tests.
 */
final class GgufQ3CalcFx {
    private GgufQ3CalcFx() {
    }

    static int quant(int index) {
        return (index * 37 + 0x63) & 0xFF;
    }

    static int highBits(int index) {
        return (index * 91 + 0xC3) & 0xFF;
    }

    static float expectedDot(float[] vector) {
        float expected = 0.0f;
        for (int superBlock = 0; superBlock < 256; superBlock += 128) {
            expected += superBlockDot(vector, superBlock);
        }
        return expected;
    }

    static float[] expectedRow() {
        float[] expected = new float[256];
        for (int superBlock = 0; superBlock < 256; superBlock += 128) {
            fillSuperBlockRow(expected, superBlock);
        }
        return expected;
    }

    private static float superBlockDot(float[] vector, int superBlock) {
        float expected = 0.0f;
        int quantBase = (superBlock / 128) * 32;
        int mask0 = 1 << (superBlock / 32);
        int mask1 = mask0 << 1;
        int mask2 = mask1 << 1;
        int mask3 = mask2 << 1;
        for (int i = 0; i < 16; i++) {
            int firstQuant = quant(quantBase + i);
            int secondQuant = quant(quantBase + 16 + i);
            int firstHigh = highBits(i);
            int secondHigh = highBits(16 + i);
            expected += value(firstQuant, firstHigh, 0, mask0) * vector[superBlock + i];
            expected += value(secondQuant, secondHigh, 0, mask0) * vector[superBlock + 16 + i];
            expected += value(firstQuant, firstHigh, 2, mask1) * vector[superBlock + 32 + i];
            expected += value(secondQuant, secondHigh, 2, mask1) * vector[superBlock + 48 + i];
            expected += value(firstQuant, firstHigh, 4, mask2) * vector[superBlock + 64 + i];
            expected += value(secondQuant, secondHigh, 4, mask2) * vector[superBlock + 80 + i];
            expected += value(firstQuant, firstHigh, 6, mask3) * vector[superBlock + 96 + i];
            expected += value(secondQuant, secondHigh, 6, mask3) * vector[superBlock + 112 + i];
        }
        return expected;
    }

    private static void fillSuperBlockRow(float[] expected, int superBlock) {
        int quantBase = (superBlock / 128) * 32;
        int mask0 = 1 << (superBlock / 32);
        int mask1 = mask0 << 1;
        int mask2 = mask1 << 1;
        int mask3 = mask2 << 1;
        for (int i = 0; i < 16; i++) {
            int firstQuant = quant(quantBase + i);
            int secondQuant = quant(quantBase + 16 + i);
            int firstHigh = highBits(i);
            int secondHigh = highBits(16 + i);
            expected[superBlock + i] = value(firstQuant, firstHigh, 0, mask0);
            expected[superBlock + 16 + i] = value(secondQuant, secondHigh, 0, mask0);
            expected[superBlock + 32 + i] = value(firstQuant, firstHigh, 2, mask1);
            expected[superBlock + 48 + i] = value(secondQuant, secondHigh, 2, mask1);
            expected[superBlock + 64 + i] = value(firstQuant, firstHigh, 4, mask2);
            expected[superBlock + 80 + i] = value(secondQuant, secondHigh, 4, mask2);
            expected[superBlock + 96 + i] = value(firstQuant, firstHigh, 6, mask3);
            expected[superBlock + 112 + i] = value(secondQuant, secondHigh, 6, mask3);
        }
    }

    private static float value(int packedQuant, int highBits, int shift, int highMask) {
        return ((packedQuant >>> shift) & 0x03) - highBias(highBits, highMask);
    }

    private static int highBias(int highBits, int highMask) {
        return (((highBits & highMask) - 1) >>> 31) << 2;
    }
}
