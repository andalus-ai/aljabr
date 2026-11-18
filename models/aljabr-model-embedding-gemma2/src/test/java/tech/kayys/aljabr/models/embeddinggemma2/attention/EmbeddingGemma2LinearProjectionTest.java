package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2LinearProjectionTest {

    @Test
    void appliesOutputMajorCheckpointWeightsAndOptionalBias() {
        float[][] input = {{2f, 3f}};
        float[][] weights = {{1f, 2f}, {-1f, 4f}};
        float[][] result = EmbeddingGemma2LinearProjection.apply(input, weights, new float[]{5f, -2f});
        assertArrayEquals(new float[]{13f, 8f}, result[0]);
        assertArrayEquals(new float[]{2f, 3f}, input[0]);
    }

    @Test
    void rejectsMalformedAndNonFiniteCheckpointWeights() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2LinearProjection.apply(new float[][]{{1f, 2f}},
                        new float[][]{{1f}}, null));
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2LinearProjection.apply(new float[][]{{1f}},
                        new float[][]{{Float.NaN}}, null));
    }
}
