package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionTokenSplitterTest {
    @Test
    void splitsFlattenedSoftTokensByValidPatchPositions() {
        // pooling kernel 2; patch axis 8 => output length 2.
        // Position width 8 / output length 2 => 2x2 patches per soft token.
        int[][][] positions = {
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}},
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}
        };
        float[][] projected = {
            {1f, 10f}, {2f, 20f}, {3f, 30f}
        };

        List<float[][]> result = EmbeddingGemma2VisionTokenSplitter.split(
                projected, positions, 8, 2);

        assertEquals(2, result.size());
        assertEquals(2, result.get(0).length);
        assertEquals(1, result.get(1).length);
        assertEquals(1f, result.get(0)[0][0]);
        assertEquals(3f, result.get(1)[0][0]);
    }

    @Test
    void rejectsProjectedTokenCountMismatch() {
        int[][][] positions = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2VisionTokenSplitter.split(
                        new float[][]{{1f, 2f}, {3f, 4f}},
                        positions, 4, 2));
    }

    @Test
    void rejectsNonSquarePatchToSoftTokenRatio() {
        int[][][] positions = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}}};
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2VisionTokenSplitter.split(
                        new float[][]{{1f, 2f}},
                        positions, 4, 2));
    }

    @Test
    void rejectsMalformedPaddingCoordinates() {
        int[][][] positions = {{{0, 0}, {-1, 0}, {1, 0}, {1, 1}}};
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2VisionTokenSplitter.split(
                        new float[][]{{1f, 2f}},
                        positions, 4, 2));
    }
}
