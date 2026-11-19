package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionRotaryEmbeddingTest {
    @Test
    void zeroPositionLeavesEveryHeadUnchanged() {
        var rope = new EmbeddingGemma2VisionRotaryEmbedding(8, 100.0);
        float[][][] input = {{{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}}};
        float[][][] output = rope.apply(input, new int[][]{{0, 0}});
        assertArrayEquals(input[0][0], output[0][0], 1e-6f);
        assertArrayEquals(new float[]{1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f},
                rope.compute(new int[][]{{0, 0}}).cosine()[0], 1e-6f);
        assertArrayEquals(new float[8], rope.compute(new int[][]{{0, 0}}).sine()[0], 1e-6f);
    }

    @Test
    void rotatesTheTwoSpatialAxesIndependently() {
        var rope = new EmbeddingGemma2VisionRotaryEmbedding(4, 100.0);
        float[][][] input = {{{1f, 0f, 1f, 0f}}};
        float[][][] output = rope.apply(input, new int[][]{{1, 2}});
        assertEquals((float) Math.cos(1), output[0][0][0], 1e-6f);
        assertEquals((float) Math.sin(1), output[0][0][1], 1e-6f);
        assertEquals((float) Math.cos(2), output[0][0][2], 1e-6f);
        assertEquals((float) Math.sin(2), output[0][0][3], 1e-6f);
    }

    @Test
    void rejectsInvalidHeadGeometryAndPositions() {
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2VisionRotaryEmbedding(6, 100.0));
        var rope = new EmbeddingGemma2VisionRotaryEmbedding(8, 100.0);
        assertThrows(IllegalArgumentException.class, () -> rope.compute(new int[][]{{-1, 0}}));
    }

    @Test
    void doesNotMutateInput() {
        var rope = new EmbeddingGemma2VisionRotaryEmbedding(4, 100.0);
        float[][][] input = {{{1f, 2f, 3f, 4f}}};
        float[] original = input[0][0].clone();
        rope.apply(input, new int[][]{{3, 5}});
        assertArrayEquals(original, input[0][0]);
    }
}
