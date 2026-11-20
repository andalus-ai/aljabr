package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
                                {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f},
                                {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f},
                                {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}
                        },
                        {
                                {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f},
                                {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f},
                                {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f},
                                {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}
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
    }

    @Test
    void rejectsProcessorMetadataWhenItDoesNotMatchValidPooledTokens() {
        var tower = tower();
        var processorOutput = new EmbeddingGemma2VisionProcessorOutput(
                new float[][][]{{{0.5f, 0.5f, 0.5f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}, {0f, 0f, 0f}}},
                new int[][][]{{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}},
                new int[]{2});

        assertThrows(IllegalArgumentException.class, () -> tower.encodeAndProject(processorOutput));
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

    private static EmbeddingGemma2VisionTower tower() {
        float[][] zeros = new float[4][4];
        var patch = new EmbeddingGemma2VisionPatchEmbedder.Weights(
                new float[4][3], new float[2][4][4]);
        var attention = new EmbeddingGemma2VisionAttention.Weights(
                zeros, zeros, zeros, zeros,
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
