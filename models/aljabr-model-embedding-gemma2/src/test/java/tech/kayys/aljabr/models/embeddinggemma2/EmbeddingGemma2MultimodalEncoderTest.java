package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2MultimodalEncoderTest {
    @Test
    void rejectsMaskLengthMismatchBeforeVisionExecution() {
        var backbone = testBackbone();
        assertThrows(IllegalArgumentException.class, () ->
                backbone.validateSequenceMetadata(3, new boolean[] {true, true}, new int[] {0, 1, 2}));
    }

    @Test
    void rejectsPositionLengthMismatchAndOutOfRangePositionBeforeVisionExecution() {
        var backbone = testBackbone();
        assertThrows(IllegalArgumentException.class, () ->
                backbone.validateSequenceMetadata(3, null, new int[] {0, 1}));
        assertThrows(IllegalArgumentException.class, () ->
                backbone.validateSequenceMetadata(3, null, new int[] {0, 1, 999}));
    }

    @Test
    void rejectsSequenceLengthOutsideConfiguredLimitBeforeVisionExecution() {
        var backbone = testBackbone();
        assertThrows(IllegalArgumentException.class, () ->
                backbone.validateSequenceMetadata(0, null, new int[0]));
    }

    private static EmbeddingGemma2TextBackbone testBackbone() {
        return EmbeddingGemma2TestFixtures.tinyBackbone();
    }

    @Test
    void validatesPlaceholderBudgetAgainstProcessorTokenCounts() {
        assertDoesNotThrow(() -> EmbeddingGemma2MultimodalEncoder.validateImagePlaceholderCount(
                new int[] {7, 99, 99, 8, 99}, 99, new int[] {2, 1}));
    }

    @Test
    void rejectsPlaceholderBudgetMismatchBeforeVisionExecution() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalEncoder.validateImagePlaceholderCount(
                        new int[] {7, 99, 8}, 99, new int[] {2}));
    }

    @Test
    void rejectsNonPositiveImageSoftTokenCounts() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalEncoder.validateImagePlaceholderCount(
                        new int[] {99}, 99, new int[] {0}));
    }

    @Test
    void rejectsMissingInputsAndInvalidPlaceholderIds() {
        assertThrows(NullPointerException.class, () ->
                EmbeddingGemma2MultimodalEncoder.validateImagePlaceholderCount(null, 99, new int[] {1}));
        assertThrows(NullPointerException.class, () ->
                EmbeddingGemma2MultimodalEncoder.validateImagePlaceholderCount(new int[] {99}, 99, null));
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalEncoder.validateImagePlaceholderCount(new int[] {99}, -1, new int[] {1}));
    }
}
