package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2MultimodalInputAssemblerTest {
    @Test
    void replacesImageSlotsInPromptOrderAndPreservesImageBoundaries() {
        float[][] text = {
                {1f, 1f}, {2f, 2f}, {3f, 3f}, {4f, 4f}, {5f, 5f}, {6f, 6f}
        };
        int[] ids = {10, 99, 99, 11, 99, 12};
        List<float[][]> images = List.of(
                new float[][]{{20f, 21f}, {22f, 23f}},
                new float[][]{{30f, 31f}});

        float[][] merged = EmbeddingGemma2MultimodalInputAssembler.replaceImagePlaceholders(
                text, ids, 99, images);

        assertArrayEquals(new float[]{1f, 1f}, merged[0], 0f);
        assertArrayEquals(new float[]{20f, 21f}, merged[1], 0f);
        assertArrayEquals(new float[]{22f, 23f}, merged[2], 0f);
        assertArrayEquals(new float[]{4f, 4f}, merged[3], 0f);
        assertArrayEquals(new float[]{30f, 31f}, merged[4], 0f);
        assertArrayEquals(new float[]{6f, 6f}, merged[5], 0f);
        assertArrayEquals(new float[]{2f, 2f}, text[1], 0f);
    }

    @Test
    void rejectsPlaceholderFeatureCountMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalInputAssembler.replaceImagePlaceholders(
                        new float[][]{{1f, 1f}, {2f, 2f}},
                        new int[]{99, 10},
                        99,
                        List.of(new float[][]{{3f, 4f}, {5f, 6f}})));
    }

    @Test
    void rejectsWrongFeatureWidthAndNonFiniteFeatures() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalInputAssembler.replaceImagePlaceholders(
                        new float[][]{{1f, 1f}},
                        new int[]{99},
                        99,
                        List.of(new float[][]{{2f}})));
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalInputAssembler.replaceImagePlaceholders(
                        new float[][]{{1f, 1f}},
                        new int[]{99},
                        99,
                        List.of(new float[][]{{Float.NaN, 2f}})));
    }

    @Test
    void rejectsMalformedTextEmbeddingsAndNegativeImageTokenId() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalInputAssembler.replaceImagePlaceholders(
                        new float[][]{{1f}, {2f, 2f}},
                        new int[]{10, 99},
                        99,
                        List.of(new float[][]{{3f}})));
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalInputAssembler.replaceImagePlaceholders(
                        new float[][]{{1f}},
                        new int[]{10},
                        -1,
                        List.of()));
    }
}
