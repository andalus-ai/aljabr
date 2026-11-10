package tech.kayys.aljabr.gguf.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Shared assertions for ternary-quantized block lane patterns.
 */
final class GgufTQFx {
    private GgufTQFx() {
    }

    static void assertTQ1Pattern(
            float[] row,
            float first,
            float second,
            float third,
            float fourth,
            float fifth) {
        int offset = 0;
        float[] values = {first, second, third, fourth, fifth};
        for (float value : values) {
            for (int i = 0; i < 32; i++) {
                assertEquals(value, row[offset++], 0.0f);
            }
        }
        for (float value : values) {
            for (int i = 0; i < 16; i++) {
                assertEquals(value, row[offset++], 0.0f);
            }
        }
        for (int lane = 0; lane < 4; lane++) {
            for (int i = 0; i < 4; i++) {
                assertEquals(values[lane], row[offset++], 0.0f);
            }
        }
        assertEquals(row.length, offset);
    }

    static void assertTQ2Pattern(float[] row, float first, float second, float third, float fourth) {
        for (int group = 0; group < 2; group++) {
            int base = group * 128;
            for (int i = 0; i < 32; i++) {
                assertEquals(first, row[base + i], 0.0f);
                assertEquals(second, row[base + 32 + i], 0.0f);
                assertEquals(third, row[base + 64 + i], 0.0f);
                assertEquals(fourth, row[base + 96 + i], 0.0f);
            }
        }
    }
}
