package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2MultimodalEmbedderTest {
    @Test
    void normalizesWithoutLearnedScaleThenProjectsToTextWidth() {
        EmbeddingGemma2MultimodalEmbedder embedder = new EmbeddingGemma2MultimodalEmbedder(
                2, 3, 1.0e-6f,
                new EmbeddingGemma2MultimodalEmbedder.Weights(new float[][]{
                    {1f, 0f},
                    {0f, 1f},
                    {1f, 1f}
                }));

        float[][] result = embedder.project(new float[][]{{3f, 4f}, {0f, 2f}});

        assertEquals(2, result.length);
        assertEquals(3, result[0].length);
        assertEquals(3f / (float) Math.sqrt(12.5f), result[0][0], 1.0e-5f);
        assertEquals(4f / (float) Math.sqrt(12.5f), result[0][1], 1.0e-5f);
        assertEquals(result[0][0] + result[0][1], result[0][2], 1.0e-5f);
    }

    @Test
    void preservesImageBoundariesWhenProjectingMultipleImages() {
        EmbeddingGemma2MultimodalEmbedder embedder = new EmbeddingGemma2MultimodalEmbedder(
                2, 2, 1.0e-6f,
                new EmbeddingGemma2MultimodalEmbedder.Weights(new float[][]{
                    {1f, 0f}, {0f, 1f}
                }));

        List<float[][]> result = embedder.projectImages(List.of(
                new float[][]{{1f, 0f}, {0f, 1f}},
                new float[][]{{2f, 2f}}));

        assertEquals(2, result.size());
        assertEquals(2, result.get(0).length);
        assertEquals(1, result.get(1).length);
    }

    @Test
    void rejectsProjectionShapeMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2MultimodalEmbedder(
                        2, 3, 1.0e-6f,
                        new EmbeddingGemma2MultimodalEmbedder.Weights(new float[][]{{1f, 0f}})));
    }

    @Test
    void rejectsMalformedVisionTokenWidth() {
        EmbeddingGemma2MultimodalEmbedder embedder = new EmbeddingGemma2MultimodalEmbedder(
                2, 2, 1.0e-6f,
                new EmbeddingGemma2MultimodalEmbedder.Weights(new float[][]{
                    {1f, 0f}, {0f, 1f}
                }));

        assertThrows(IllegalArgumentException.class,
                () -> embedder.project(new float[][]{{1f, 2f, 3f}}));
    }

    @Test
    void rejectsEmptyImageCollection() {
        EmbeddingGemma2MultimodalEmbedder embedder = new EmbeddingGemma2MultimodalEmbedder(
                2, 2, 1.0e-6f,
                new EmbeddingGemma2MultimodalEmbedder.Weights(new float[][]{
                    {1f, 0f}, {0f, 1f}
                }));

        assertThrows(IllegalArgumentException.class, () -> embedder.projectImages(List.of()));
    }
}
