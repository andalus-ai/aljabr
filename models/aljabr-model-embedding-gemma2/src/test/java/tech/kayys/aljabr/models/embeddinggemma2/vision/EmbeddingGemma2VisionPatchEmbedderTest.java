package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionPatchEmbedderTest {

    @Test
    void scalesPixelsProjectsAndAddsBothAxisEmbeddings() {
        // patchSize=1 => three RGB input values, one output channel.
        var embedder = new EmbeddingGemma2VisionPatchEmbedder(
                1, 1, 4,
                new EmbeddingGemma2VisionPatchEmbedder.Weights(
                        new float[][]{{1f, 2f, 3f}},
                        new float[][][]{{{0f}, {10f}, {20f}, {30f}},
                                {{0f}, {100f}, {200f}, {300f}}}));

        float[][] result = embedder.embed(
                new float[][]{{0.5f, 0.5f, 0.5f}, {1f, 0f, 0.5f}},
                new int[][]{{1, 2}, {-1, -1}});

        // First patch: scaled RGB=(0,0,0), plus x[1]=10 and y[2]=200.
        assertEquals(210f, result[0][0], 1e-6f);
        // Padding receives no positional vector; projection still runs.
        assertEquals(-1f, result[1][0], 1e-6f);
    }

    @Test
    void rejectsHalfPaddingSentinel() {
        var embedder = new EmbeddingGemma2VisionPatchEmbedder(
                1, 1, 2,
                new EmbeddingGemma2VisionPatchEmbedder.Weights(
                        new float[][]{{1f, 1f, 1f}},
                        new float[][][]{{{0f}, {0f}}, {{0f}, {0f}}}));

        assertThrows(IllegalArgumentException.class,
                () -> embedder.embed(new float[][]{{0f, 0f, 0f}}, new int[][]{{-1, 0}}));
    }

    @Test
    void rejectsOutOfRangeCoordinates() {
        var embedder = new EmbeddingGemma2VisionPatchEmbedder(
                1, 1, 2,
                new EmbeddingGemma2VisionPatchEmbedder.Weights(
                        new float[][]{{1f, 1f, 1f}},
                        new float[][][]{{{0f}, {0f}}, {{0f}, {0f}}}));

        assertThrows(IllegalArgumentException.class,
                () -> embedder.embed(new float[][]{{0f, 0f, 0f}}, new int[][]{{2, 0}}));
    }

    @Test
    void reportsConfiguredDimensions() {
        var embedder = new EmbeddingGemma2VisionPatchEmbedder(
                1, 1, 2,
                new EmbeddingGemma2VisionPatchEmbedder.Weights(
                        new float[1][3], new float[2][2][1]));

        assertEquals(1, embedder.patchSize());
        assertEquals(1, embedder.hiddenSize());
        assertEquals(2, embedder.positionEmbeddingSize());
    }
}
