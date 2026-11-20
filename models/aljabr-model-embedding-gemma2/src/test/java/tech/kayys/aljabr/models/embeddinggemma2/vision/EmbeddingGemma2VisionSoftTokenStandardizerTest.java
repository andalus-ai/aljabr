package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionSoftTokenStandardizerTest {
    @Test
    void appliesCheckpointFeaturewiseBiasThenScale() {
        var standardizer = new EmbeddingGemma2VisionSoftTokenStandardizer(
                3, new float[]{1f, 2f, -1f}, new float[]{2f, 0.5f, -3f});

        float[][] result = standardizer.standardize(new float[][]{{3f, 6f, 2f}});

        assertArrayEquals(new float[]{4f, 2f, -9f}, result[0], 1e-6f);
    }

    @Test
    void doesNotMutateInputOrCheckpointArrays() {
        float[] bias = {1f, 2f};
        float[] scale = {3f, 4f};
        var standardizer = new EmbeddingGemma2VisionSoftTokenStandardizer(2, bias, scale);
        float[][] input = {{5f, 6f}};

        float[][] result = standardizer.standardize(input);

        assertArrayEquals(new float[]{5f, 6f}, input[0], 1e-6f);
        assertArrayEquals(new float[]{1f, 2f}, bias, 1e-6f);
        assertArrayEquals(new float[]{3f, 4f}, scale, 1e-6f);
        assertArrayEquals(new float[]{12f, 16f}, result[0], 1e-6f);
    }

    @Test
    void rejectsMismatchedDimensionsAndNonFiniteValues() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionSoftTokenStandardizer(
                        2, new float[]{0f}, new float[]{1f, 1f}));
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionSoftTokenStandardizer(
                        1, new float[]{Float.NaN}, new float[]{1f}));
        var standardizer = new EmbeddingGemma2VisionSoftTokenStandardizer(
                1, new float[]{0f}, new float[]{1f});
        assertThrows(IllegalArgumentException.class, () ->
                standardizer.standardize(new float[][]{{Float.POSITIVE_INFINITY}}));
    }

    @Test
    void rejectsNullAndRaggedInputs() {
        var standardizer = new EmbeddingGemma2VisionSoftTokenStandardizer(
                2, new float[]{0f, 0f}, new float[]{1f, 1f});
        assertThrows(NullPointerException.class, () -> standardizer.standardize(null));
        assertThrows(IllegalArgumentException.class, () ->
                standardizer.standardize(new float[][]{{1f}, {2f, 3f}}));
    }

    @Test
    void supportsEmptyTokenSequence() {
        var standardizer = new EmbeddingGemma2VisionSoftTokenStandardizer(
                2, new float[]{0f, 0f}, new float[]{1f, 1f});
        assertEquals(0, standardizer.standardize(new float[0][2]).length);
    }
}
