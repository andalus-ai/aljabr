package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2SequenceRmsNormTest {
    @Test
    void normalizesEachSequenceRowWithCheckpointScale() {
        float[][] result = EmbeddingGemma2SequenceRmsNorm.apply(
                new float[][]{{3f, 4f}}, new float[]{2f, 0.5f}, 0f + 1e-6f);
        assertEquals(1.697056f, result[0][0], 1e-5f);
        assertEquals(0.565685f, result[0][1], 1e-5f);
    }
}
