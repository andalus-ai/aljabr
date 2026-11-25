package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;

class EmbeddingGemma2VisionProcessorBatchTest {
    @Test
    void executesMultipleProcessorImagesAndPreservesEachSoftTokenBoundary() {
        var tower = tower();
        var processorOutput = new EmbeddingGemma2VisionProcessorOutput(
                new float[][][]{
                        {
                                {0.6f, 0.4f, 0.7f}, {0.7f, 0.3f, 0.5f},
                                {0.4f, 0.8f, 0.2f}, {0.9f, 0.1f, 0.6f},
                                {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}
                        },
                        {
                                {0.55f, 0.45f, 0.65f}, {0.65f, 0.35f, 0.55f},
                                {0.45f, 0.75f, 0.25f}, {0.85f, 0.15f, 0.55f},
                                {0.35f, 0.65f, 0.75f}, {0.75f, 0.25f, 0.35f},
                                {0.25f, 0.85f, 0.45f}, {0.95f, 0.05f, 0.65f}
                        }
                },
                new int[][][]{
                        {
                                {0, 0}, {1, 0}, {0, 1}, {1, 1},
                                {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}
                        },
                        {
                                {0, 0}, {1, 0}, {2, 0}, {3, 0},
                                {0, 1}, {1, 1}, {2, 1}, {3, 1}
                        }
                },
                new int[]{1, 2});

        List<float[][]> features = tower.encodeAndProject(processorOutput);

        assertEquals(2, features.size());
        assertEquals(1, features.get(0).length);
        assertEquals(2, features.get(1).length);
        assertEquals(4, features.get(0)[0].length);
        assertEquals(4, features.get(1)[0].length);
        assertTrue(java.util.Arrays.stream(features.get(0))
                .flatMapToDouble(row -> java.util.Arrays.stream(row).asDoubleStream())
                .anyMatch(value -> Math.abs(value) > 1e-7),
                "non-zero fixture must exercise numerical feature preservation");

        // Reference the previous full padded-axis execution. Valid tokens must
        // remain numerically unchanged when the tower trims the padded suffix before attention.
        float[][][] paddedPixels = processorOutput.pixelValues();
        int[][][] paddedPositions = processorOutput.imagePositionIds();
        int paddedOutputLength = processorOutput.paddedPatchCount() / 4;
        for (int image = 0; image < features.size(); image++) {
            float[][] untrimmed = tower.encodeAndProject(
                    paddedPixels[image], paddedPositions[image], paddedOutputLength, null);
            assertMatrixEquals(untrimmed, features.get(image));
        }
    }

    @Test
    void rejectsProcessorMetadataWhenItDoesNotMatchValidPooledTokens() {
        var tower = tower();
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionProcessorOutput(
                        new float[][][]{{{0.5f, 0.5f, 0.5f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}}},
                        new int[][][]{{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}},
                        new int[]{2}));
    }

    @Test
    void processorOutputDefensivelyCopiesInputAndAccessors() {
        float[][][] pixels = {{{0.25f, 0.5f, 0.75f}}};
        int[][][] positions = {{{0, 0}}};
        int[] counts = {1};
        var output = new EmbeddingGemma2VisionProcessorOutput(pixels, positions, counts);

        pixels[0][0][0] = 0.9f;
        positions[0][0][0] = 3;
        counts[0] = 8;
        float[][][] returnedPixels = output.pixelValues();
        returnedPixels[0][0][0] = 0.1f;

        assertEquals(0.25f, output.pixelValues()[0][0][0], 1e-6f);
        assertEquals(0, output.imagePositionIds()[0][0][0]);
        assertEquals(1, output.numSoftTokensPerImage()[0]);
    }

    @Test
    void rejectsProcessorBatchWithInvalidPaddingOrNonFinitePixels() {
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionProcessorOutput(
                        new float[][][]{{{0.5f, 0.5f, 0.5f}}},
                        new int[][][]{{{-1, 0}}},
                        new int[]{1}));
        assertThrows(IllegalArgumentException.class, () ->
                new EmbeddingGemma2VisionProcessorOutput(
                        new float[][][]{{{Float.NaN, 0.5f, 0.5f}}},
                        new int[][][]{{{0, 0}}},
                        new int[]{1}));
    }

    private static float[][] identity(int size) {
        float[][] result = new float[size][size];
        for (int index = 0; index < size; index++) result[index][index] = 1f;
        return result;
    }

    private static void assertMatrixEquals(float[][] expected, float[][] actual) {
        assertEquals(expected.length, actual.length, "token count");
        for (int token = 0; token < expected.length; token++) {
            assertEquals(expected[token].length, actual[token].length, "hidden size");
            for (int dimension = 0; dimension < expected[token].length; dimension++) {
                assertEquals(expected[token][dimension], actual[token][dimension], 1e-6f,
                        "token " + token + ", dimension " + dimension);
            }
        }
    }

    private static EmbeddingGemma2VisionTower tower() {
        float[][] zeros = new float[4][4];
        var patch = new EmbeddingGemma2VisionPatchEmbedder.Weights(
                new float[][]{{1f, 0f, 0f}, {0f, 1f, 0f}, {0f, 0f, 1f}, {1f, 1f, 1f}},
                new float[2][4][4]);
        var attention = new EmbeddingGemma2VisionAttention.Weights(
                zeros, zeros, identity(4), identity(4),
                new float[]{1f, 1f, 1f, 1f}, new float[]{1f, 1f, 1f, 1f});
        var mlp = new EmbeddingGemma2Mlp.Weights(zeros, zeros, zeros);
        var norms = new EmbeddingGemma2VisionEncoderLayer.Weights(
                new float[]{1f, 1f, 1f, 1f},
                new float[]{1f, 1f, 1f, 1f},
                new float[]{1f, 1f, 1f, 1f},
                new float[]{1f, 1f, 1f, 1f});
        var weights = new EmbeddingGemma2VisionCheckpointMapper.Weights(
                patch,
                List.of(new EmbeddingGemma2VisionCheckpointMapper.LayerWeights(attention, mlp, norms)),
                null, null, null);
        var spec = new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                4, 4, 1, 1, 1, 4, 1, 4, false);
        var projection = new float[][]{
                {1f, 0f, 0f, 0f},
                {0f, 1f, 0f, 0f},
                {0f, 0f, 1f, 0f},
                {0f, 0f, 0f, 1f}
        };
        var embedder = new EmbeddingGemma2MultimodalEmbedder(
                4, 4, 1e-6f, new EmbeddingGemma2MultimodalEmbedder.Weights(projection));
        return EmbeddingGemma2VisionTower.fromCheckpoint(spec, weights, embedder, 1e-6f, 100d, 2);
    }
}
