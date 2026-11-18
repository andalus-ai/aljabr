package tech.kayys.aljabr.models.embeddinggemma2.ple;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2PerLayerEmbeddingProjectorTest {

    @Test
    void projectsAndNormalizesEveryTokenLayerSlice() {
        float[][] inputs = {{1f, 0f}, {0f, 2f}};
        float[][] weights = new float[4][2];
        weights[0][0] = 1f;
        weights[1][1] = 1f;
        weights[2][1] = 2f;
        weights[3][0] = 3f;
        float[] normScale = {1f, 1f};

        float[][][] result = EmbeddingGemma2PerLayerEmbeddingProjector.project(
                inputs, weights, normScale, 2, 2, 1e-6f);

        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertEquals(2, result[0][0].length);
        assertEquals(1f, norm(result[0][0]), 1e-5f);
        assertEquals(1f, norm(result[0][1]), 1e-5f);
        assertEquals(1f, norm(result[1][0]), 1e-5f);
        assertEquals(1f, norm(result[1][1]), 1e-5f);
    }

    @Test
    void appliesHiddenSizeScalingBeforeReshapeAndNorm() {
        float[][] input = {{3f, 4f}};
        float[][] weights = {{1f, 0f}, {0f, 1f}};
        float[][][] result = EmbeddingGemma2PerLayerEmbeddingProjector.project(
                input, weights, new float[] {1f, 1f}, 1, 2, 1e-6f);
        assertEquals(0.6f, result[0][0][0], 1e-5f);
        assertEquals(0.8f, result[0][0][1], 1e-5f);
    }

    @Test
    void rejectsInvalidDimensionsAndNonFiniteValues() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2PerLayerEmbeddingProjector.project(
                        new float[][] {{1f}}, new float[3][1], new float[] {1f}, 1, 2, 1e-6f));
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2PerLayerEmbeddingProjector.project(
                        new float[][] {{Float.NaN}}, new float[][] {{1f}}, new float[] {1f}, 1, 1, 1e-6f));
    }

    private static float norm(float[] vector) {
        double sum = 0;
        for (float value : vector) sum += (double) value * value;
        return (float) Math.sqrt(sum);
    }
}
