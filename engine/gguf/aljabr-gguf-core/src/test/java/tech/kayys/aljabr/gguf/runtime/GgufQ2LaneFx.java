package tech.kayys.aljabr.gguf.runtime;

/**
 * Q2_K lane-order expectation math shared by raw and prepared matvec tests.
 */
final class GgufQ2LaneFx {
    private GgufQ2LaneFx() {
    }

    static int quant(int index) {
        return (index * 43 + 0x39) & 0xFF;
    }

    static float expectedDot(float[] vector, boolean hasMins) {
        float expected = 0.0f;
        for (int superBlock = 0; superBlock < 256; superBlock += 128) {
            int quantBase = (superBlock / 128) * 32;
            for (int i = 0; i < 16; i++) {
                int firstQuant = quant(quantBase + i);
                int secondQuant = quant(quantBase + 16 + i);
                expected += value(firstQuant, 0, hasMins) * vector[superBlock + i];
                expected += value(secondQuant, 0, hasMins) * vector[superBlock + 16 + i];
                expected += value(firstQuant, 2, hasMins) * vector[superBlock + 32 + i];
                expected += value(secondQuant, 2, hasMins) * vector[superBlock + 48 + i];
                expected += value(firstQuant, 4, hasMins) * vector[superBlock + 64 + i];
                expected += value(secondQuant, 4, hasMins) * vector[superBlock + 80 + i];
                expected += value(firstQuant, 6, hasMins) * vector[superBlock + 96 + i];
                expected += value(secondQuant, 6, hasMins) * vector[superBlock + 112 + i];
            }
        }
        return expected;
    }

    static float[] expectedRow(boolean hasMins) {
        float[] expected = new float[256];
        for (int superBlock = 0; superBlock < 256; superBlock += 128) {
            int quantBase = (superBlock / 128) * 32;
            for (int i = 0; i < 16; i++) {
                int firstQuant = quant(quantBase + i);
                int secondQuant = quant(quantBase + 16 + i);
                expected[superBlock + i] = value(firstQuant, 0, hasMins);
                expected[superBlock + 16 + i] = value(secondQuant, 0, hasMins);
                expected[superBlock + 32 + i] = value(firstQuant, 2, hasMins);
                expected[superBlock + 48 + i] = value(secondQuant, 2, hasMins);
                expected[superBlock + 64 + i] = value(firstQuant, 4, hasMins);
                expected[superBlock + 80 + i] = value(secondQuant, 4, hasMins);
                expected[superBlock + 96 + i] = value(firstQuant, 6, hasMins);
                expected[superBlock + 112 + i] = value(secondQuant, 6, hasMins);
            }
        }
        return expected;
    }

    private static float value(int packedQuant, int shift, boolean hasMins) {
        int value = (packedQuant >>> shift) & 0x03;
        return hasMins ? 2.0f * value - 1.0f : value;
    }
}
