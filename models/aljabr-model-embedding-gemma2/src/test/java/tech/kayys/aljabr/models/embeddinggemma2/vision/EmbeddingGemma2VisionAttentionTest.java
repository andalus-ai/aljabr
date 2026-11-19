package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionAttentionTest {
    @Test
    void performsBidirectionalAttentionAndMasksPaddingKeys() {
        float[][] identity = identity(4);
        float[] normScale = {1f, 1f, 1f, 1f};
        var attention = new EmbeddingGemma2VisionAttention(
                4, 1, 1, 4, 1e-6f, 100.0,
                new EmbeddingGemma2VisionAttention.Weights(
                        identity, identity, identity, identity, normScale, normScale));

        float[][] input = {
                {1f, 0f, 0f, 0f},
                {0f, 0f, 1f, 0f}
        };
        float[][] output = attention.compute(input,
                new int[][]{{0, 0}, {-1, -1}}, new boolean[]{true, false});

        assertEquals(2, output.length);
        assertArrayEquals(new float[]{1f, 0f, 0f, 0f}, output[0], 2e-5f);
        assertArrayEquals(new float[]{1f, 0f, 0f, 0f}, output[1], 2e-5f);
    }

    @Test
    void rejectsAnAllMaskedKeySequence() {
        float[][] identity = identity(4);
        float[] scale = {1f, 1f, 1f, 1f};
        var attention = new EmbeddingGemma2VisionAttention(
                4, 1, 1, 4, 1e-6f, 100.0,
                new EmbeddingGemma2VisionAttention.Weights(
                        identity, identity, identity, identity, scale, scale));
        assertThrows(IllegalArgumentException.class,
                () -> attention.compute(new float[][]{{1f, 0f, 0f, 0f}},
                        new int[][]{{0, 0}}, new boolean[]{false}));
    }

    @Test
    void rejectsIncompatibleProjectionShapes() {
        float[][] identity = identity(4);
        float[] scale = {1f, 1f, 1f, 1f};
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2VisionAttention(
                        4, 1, 1, 4, 1e-6f, 100.0,
                        new EmbeddingGemma2VisionAttention.Weights(
                                identity, identity, identity, new float[3][4], scale, scale)));
    }

    private static float[][] identity(int size) {
        float[][] result = new float[size][size];
        for (int i = 0; i < size; i++) result[i][i] = 1f;
        return result;
    }
}
