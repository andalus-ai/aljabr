package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImageProcessor.RgbImage;

class EmbeddingGemma2ImageEmbeddingRunnerTest {

    private static RgbImage image() {
        return RgbImage.fromRgb8(1, 1, new byte[] {10, 20, 30});
    }

    @Test
    void requestDefaultsPositionsAndDefensivelyCopiesMutableInputs() {
        int[] ids = {7, 99, 8};
        boolean[] mask = {true, true, false};
        var request = new EmbeddingGemma2ImageEmbeddingRunner.Request(
                ids, List.of(image()), mask, null, 256);

        ids[0] = 100;
        mask[0] = false;

        assertArrayEquals(new int[] {7, 99, 8}, request.inputIds());
        assertArrayEquals(new boolean[] {true, true, false}, request.validTokens());
        assertArrayEquals(new int[] {0, 1, 2}, request.positionIds());
        assertEquals(256, request.outputDimension());
    }

    @Test
    void requestAcceptsExplicitPositionsAndNativeDimension() {
        var request = new EmbeddingGemma2ImageEmbeddingRunner.Request(
                new int[] {1, 99}, List.of(image()), null, new int[] {3, 4}, 768);

        assertArrayEquals(new int[] {3, 4}, request.positionIds());
        assertEquals(768, request.outputDimension());
    }

    @Test
    void requestRejectsMissingImagesAndEmptyTokens() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2ImageEmbeddingRunner.Request(
                        new int[] {1}, List.of(), null, null, 768));
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2ImageEmbeddingRunner.Request(
                        new int[0], List.of(image()), null, null, 768));
    }

    @Test
    void requestRejectsMismatchedMasksPositionsAndUnsupportedDimensions() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2ImageEmbeddingRunner.Request(
                        new int[] {1, 2}, List.of(image()), new boolean[] {true}, null, 768));
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2ImageEmbeddingRunner.Request(
                        new int[] {1, 2}, List.of(image()), null, new int[] {0}, 768));
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2ImageEmbeddingRunner.Request(
                        new int[] {1}, List.of(image()), null, null, 384));
    }
}
