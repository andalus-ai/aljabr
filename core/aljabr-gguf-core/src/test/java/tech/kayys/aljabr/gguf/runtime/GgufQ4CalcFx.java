package tech.kayys.aljabr.gguf.runtime;

/**
 * Q4_K lane-order expectation math shared by raw and prepared matrix tests.
 */
final class GgufQ4CalcFx {
    private GgufQ4CalcFx() {
    }

    static int packedQuant(int index) {
        int low = index & 0x0F;
        int high = 15 - low;
        return low | (high << 4);
    }

    static float expectedDot(float[] vector, boolean hasMins) {
        float expected = 0.0f;
        for (int superBlock = 0; superBlock < 256; superBlock += 64) {
            int quantBase = (superBlock / 64) * 32;
            for (int i = 0; i < 32; i++) {
                expected += lowValue(quantBase, i, hasMins) * vector[superBlock + i];
                expected += highValue(quantBase, i, hasMins) * vector[superBlock + 32 + i];
            }
        }
        return expected;
    }

    static float[] expectedRow(boolean hasMins) {
        float[] expected = new float[256];
        for (int superBlock = 0; superBlock < 256; superBlock += 64) {
            int quantBase = (superBlock / 64) * 32;
            for (int i = 0; i < 32; i++) {
                expected[superBlock + i] = lowValue(quantBase, i, hasMins);
                expected[superBlock + 32 + i] = highValue(quantBase, i, hasMins);
            }
        }
        return expected;
    }

    private static float lowValue(int quantBase, int index, boolean hasMins) {
        int low = (quantBase + index) & 0x0F;
        return hasMins ? low - 1.0f : low;
    }

    private static float highValue(int quantBase, int index, boolean hasMins) {
        int high = 15 - ((quantBase + index) & 0x0F);
        return hasMins ? high - 1.0f : high;
    }
}
