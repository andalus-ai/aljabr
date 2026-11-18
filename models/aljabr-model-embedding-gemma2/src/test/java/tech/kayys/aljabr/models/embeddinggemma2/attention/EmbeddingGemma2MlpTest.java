package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2MlpTest {
    @Test
    void computesGatedGeluMlpUsingCheckpointMatrices() {
        var mlp = new EmbeddingGemma2Mlp(2, 2,
                new EmbeddingGemma2Mlp.Weights(
                        new float[][]{{1f, 0f}, {0f, 1f}},
                        new float[][]{{1f, 0f}, {0f, 1f}},
                        new float[][]{{1f, 0f}, {0f, 1f}}));
        float[][] output = mlp.compute(new float[][]{{1f, -1f}});
        assertEquals(EmbeddingGemma2Mlp.geluTanh(1f), output[0][0], 1e-6f);
        assertEquals(-EmbeddingGemma2Mlp.geluTanh(-1f), output[0][1], 1e-6f);
    }

    @Test
    void rejectsIncorrectCheckpointShapes() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2Mlp(2, 3,
                        new EmbeddingGemma2Mlp.Weights(
                                new float[][]{{1f, 0f}}, new float[][]{{1f, 0f}},
                                new float[][]{{1f, 0f}})));
    }
}
