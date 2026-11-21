package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionProcessorOutputTest {

    @Test
    void acceptsRowMajorRectangularGridWithContiguousPaddingSuffix() {
        var output = new EmbeddingGemma2VisionProcessorOutput(
                new float[1][8][2],
                new int[][][]{
                        {{0, 0}, {1, 0}, {0, 1}, {1, 1},
                                {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}
                },
                new int[]{1});

        assertEquals(1, output.imageCount());
        assertEquals(8, output.paddedPatchCount());
        assertEquals(1, output.numSoftTokensPerImage()[0]);
    }

    @Test
    void rejectsSoftTokenCountThatDoesNotMatchSquarePoolingGeometry() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionProcessorOutput(
                        new float[1][8][1],
                        new int[][][]{
                                {{0, 0}, {1, 0}, {0, 1}, {1, 1},
                                        {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}
                        },
                        new int[]{2}));
    }

    @Test
    void rejectsValidPatchesAfterPaddingHasStarted() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionProcessorOutput(
                        new float[1][4][1],
                        new int[][][]{
                                {{0, 0}, {1, 0}, {-1, -1}, {0, 1}}
                        },
                        new int[]{1}));
    }

    @Test
    void rejectsNonRowMajorOrNonRectangularPatchPositions() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionProcessorOutput(
                        new float[1][4][1],
                        new int[][][]{
                                {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
                        },
                        new int[]{1}));
    }

    @Test
    void rejectsPaddedAxisThatCannotRepresentTheInferredPoolingArea() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionProcessorOutput(
                        new float[1][6][1],
                        new int[][][]{
                                {{0, 0}, {1, 0}, {0, 1}, {1, 1},
                                        {-1, -1}, {-1, -1}}
                        },
                        new int[]{1}));
    }
}
