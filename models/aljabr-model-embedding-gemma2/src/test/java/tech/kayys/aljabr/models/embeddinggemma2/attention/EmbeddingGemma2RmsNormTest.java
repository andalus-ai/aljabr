package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2RmsNormTest {

    @Test
    void normalizesEachTokenAndHeadWithOptionalLearnedScale() {
        float[][][] input = {{{3f, 4f}}, {{0f, 2f}}};
        float[][][] output = EmbeddingGemma2RmsNorm.apply(input, new float[]{2f, 0.5f}, 1e-6f);
        assertEquals(1.697056f, output[0][0][0], 1e-5f);
        assertEquals(0.565685f, output[0][0][1], 1e-5f);
        assertEquals(0f, output[1][0][0], 1e-6f);
        assertEquals(0.7071f, output[1][0][1], 1e-3f);
        assertEquals(3f, input[0][0][0]);
    }

    @Test
    void supportsUnscaledValueNorm() {
        float[][][] output = EmbeddingGemma2RmsNorm.apply(new float[][][]{{{3f, 4f}}}, null, 1e-6f);
        assertEquals(0.8485281f, output[0][0][0], 1e-5f);
        assertEquals(1.1313708f, output[0][0][1], 1e-5f);
    }

    @Test
    void rejectsInvalidEpsilonAndNonFiniteValues() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2RmsNorm.apply(new float[][][]{{{1f}}}, null, 0f));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2RmsNorm.apply(new float[][][]{{{Float.NaN}}}, null, 1e-6f));
    }
}
