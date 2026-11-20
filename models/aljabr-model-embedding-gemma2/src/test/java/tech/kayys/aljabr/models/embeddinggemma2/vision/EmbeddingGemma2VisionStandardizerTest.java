package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionStandardizerTest {
    @Test
    void appliesFeatureWiseBiasThenScale() {
        var standardizer = new EmbeddingGemma2VisionStandardizer(
                3,
                new EmbeddingGemma2VisionStandardizer.Weights(
                        new float[]{1f, 2f, -1f},
                        new float[]{2f, 0.5f, -3f}));

        float[][] output = standardizer.apply(new float[][]{
                {3f, 6f, 2f},
                {-1f, 4f, 0f}
        });

        assertArrayEquals(new float[]{4f, 2f, -9f}, output[0], 1e-6f);
        assertArrayEquals(new float[]{-4f, 1f, -3f}, output[1], 1e-6f);
    }

    @Test
    void doesNotMutateInputOrCheckpointWeights() {
        float[] bias = {1f, 2f};
        float[] scale = {2f, 3f};
        var standardizer = new EmbeddingGemma2VisionStandardizer(
                2, new EmbeddingGemma2VisionStandardizer.Weights(bias, scale));
        float[][] input = {{4f, 5f}};

        standardizer.apply(input);
        bias[0] = 100f;
        scale[0] = 100f;
        input[0][0] = 200f;

        assertArrayEquals(new float[]{6f, 9f},
                standardizer.apply(new float[][]{{4f, 5f}})[0], 1e-6f);
    }

    @Test
    void rejectsMalformedAndNonFiniteInputsAndWeights() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionStandardizer(
                        2, new EmbeddingGemma2VisionStandardizer.Weights(
                                new float[]{0f}, new float[]{1f, 1f})));
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionStandardizer(
                        1, new EmbeddingGemma2VisionStandardizer.Weights(
                                new float[]{Float.NaN}, new float[]{1f})));

        var standardizer = new EmbeddingGemma2VisionStandardizer(
                2, new EmbeddingGemma2VisionStandardizer.Weights(
                        new float[]{0f, 0f}, new float[]{1f, 1f}));
        assertThrows(IllegalArgumentException.class,
                () -> standardizer.apply(new float[][]{{1f}}));
        assertThrows(IllegalArgumentException.class,
                () -> standardizer.apply(new float[][]{{Float.POSITIVE_INFINITY, 0f}}));
    }

    @Test
    void rejectsEmptySequence() {
        var standardizer = new EmbeddingGemma2VisionStandardizer(
                1, new EmbeddingGemma2VisionStandardizer.Weights(
                        new float[]{0f}, new float[]{1f}));
        assertThrows(IllegalArgumentException.class, () -> standardizer.apply(new float[0][1]));
    }
}
