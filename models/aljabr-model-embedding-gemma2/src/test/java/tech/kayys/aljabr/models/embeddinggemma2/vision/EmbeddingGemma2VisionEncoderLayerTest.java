package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;

class EmbeddingGemma2VisionEncoderLayerTest {
    @Test
    void preservesResidualWhenAttentionAndMlpReturnZero() {
        float[][] zero = {{0f, 0f}, {0f, 0f}};
        var mlp = new EmbeddingGemma2Mlp.Weights(zero, zero, zero);
        float[] ones = {1f, 1f};
        var weights = new EmbeddingGemma2VisionEncoderLayer.Weights(ones, ones, ones, ones);
        var layer = new EmbeddingGemma2VisionEncoderLayer(2, 2, 1e-6f,
                (states, positions, valid) -> new float[states.length][2], mlp, weights);

        float[][] input = {{1f, 2f}, {3f, 4f}};
        float[][] result = layer.compute(input, new int[][]{{0, 0}, {1, 0}}, new boolean[]{true, true});

        assertArrayEquals(new float[]{1f, 2f}, input[0]);
        assertArrayEquals(new float[]{3f, 4f}, input[1]);
        assertArrayEquals(input[0], result[0], 1e-5f);
        assertArrayEquals(input[1], result[1], 1e-5f);
    }

    @Test
    void rejectsMalformedPositionCoordinates() {
        float[][] zero = {{0f, 0f}, {0f, 0f}};
        float[] ones = {1f, 1f};
        var layer = new EmbeddingGemma2VisionEncoderLayer(2, 2, 1e-6f,
                (states, positions, valid) -> states,
                new EmbeddingGemma2Mlp.Weights(zero, zero, zero),
                new EmbeddingGemma2VisionEncoderLayer.Weights(ones, ones, ones, ones));
        assertThrows(IllegalArgumentException.class,
                () -> layer.compute(new float[][]{{1f, 2f}}, new int[][]{{-1, 0}}, null));
        assertThrows(IllegalArgumentException.class,
                () -> layer.compute(new float[][]{{1f, 2f}, {2f, 3f}},
                        new int[][]{{-1, -1}, {-1, -1}}, null));
    }
}
