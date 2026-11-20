package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;

class EmbeddingGemma2VisionTowerTest {

    @Test
    void composesPatchEncoderPoolerAndOptionalStandardizer() {
        var spec = spec(true);
        var tower = EmbeddingGemma2VisionTower.fromCheckpoint(
                spec, weights(true), 1e-6f, 100d, 1);

        var output = tower.encode(
                new float[][]{{0f, 0f, 0f}},
                new int[][]{{0, 0}},
                1);

        assertEquals(1, output.hiddenStates().length);
        assertArrayEquals(new float[]{-2f, -2f, -2f, -2f}, output.hiddenStates()[0], 1e-5f);
        assertEquals(1, output.poolingKernelSize());
    }

    @Test
    void removesPaddedSpatialTokensBeforeReturningVisionFeatures() {
        var tower = EmbeddingGemma2VisionTower.fromCheckpoint(
                spec(false), weights(false), 1e-6f, 100d, 2);
        var output = tower.encode(
                new float[][]{
                        {0f, 0f, 0f}, {0f, 0f, 0f},
                        {0f, 0f, 0f}, {0f, 0f, 0f}
                },
                new int[][]{{0, 0}, {1, 0}, {0, 1}, {-1, -1}},
                1);
        assertEquals(1, output.hiddenStates().length);
    }

    @Test
    void projectsTowerTokensIntoTextHiddenSpaceAfterStandardization() {
        var tower = EmbeddingGemma2VisionTower.fromCheckpoint(
                spec(true), weights(true), 1e-6f, 100d, 1);
        float[][] projection = {
                {1f, 1f, 1f, 1f},
                {0.5f, 0.5f, 0.5f, 0.5f}
        };
        var embedder = new EmbeddingGemma2MultimodalEmbedder(
                4, 2, 1e-6f, new EmbeddingGemma2MultimodalEmbedder.Weights(projection));

        float[][] projected = tower.encodeAndProject(
                new float[][]{{0f, 0f, 0f}},
                new int[][]{{0, 0}},
                1,
                embedder);

        assertArrayEquals(new float[]{-4f, -2f}, projected[0], 1e-5f);
    }

    @Test
    void constructsTowerWithCheckpointMappedMultimodalProjection() {
        var tensors = java.util.Map.of(
                "embed_vision.embedding_projection.weight",
                new tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor(
                        new int[]{2, 4},
                        new float[]{1f, 1f, 1f, 1f, 0.5f, 0.5f, 0.5f, 0.5f}));
        var embedder = EmbeddingGemma2MultimodalCheckpointMapper.mapVisionEmbedder(
                4, 2, 1e-6f, tensors);
        var tower = EmbeddingGemma2VisionTower.fromCheckpoint(
                spec(true), weights(true), embedder, 1e-6f, 100d, 1);

        float[][] projected = tower.encodeAndProject(
                new float[][]{{0f, 0f, 0f}},
                new int[][]{{0, 0}},
                1,
                null);

        assertArrayEquals(new float[]{-4f, -2f}, projected[0], 1e-5f);
    }

    @Test
    void rejectsCheckpointLayerCountMismatch() {
        var spec = spec(true);
        var weights = new EmbeddingGemma2VisionCheckpointMapper.Weights(
                weights(true).patchEmbedder(), List.of(), null,
                new float[]{1f, 1f, 1f, 1f}, new float[]{1f, 1f, 1f, 1f});
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionTower.fromCheckpoint(spec, weights, 1e-6f, 100d, 1));
    }

    @Test
    void rejectsProjectionWhenNoEmbedderIsConfigured() {
        var tower = EmbeddingGemma2VisionTower.fromCheckpoint(
                spec(false), weights(false), 1e-6f, 100d, 1);
        assertThrows(IllegalStateException.class, () -> tower.encodeAndProject(
                new float[][]{{0f, 0f, 0f}}, new int[][]{{0, 0}}, 1, null));
    }

    private static EmbeddingGemma2VisionCheckpointMapper.VisionSpec spec(boolean standardize) {
        return new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                4, 4, 1, 1, 1, 4, 1, 4, standardize);
    }

    private static EmbeddingGemma2VisionCheckpointMapper.Weights weights(boolean standardize) {
        float[][] zeros = new float[4][4];
        float[][] mlpGate = new float[4][4];
        float[][] mlpUp = new float[4][4];
        float[][] mlpDown = new float[4][4];
        float[][][] positions = new float[2][4][4];
        var patch = new EmbeddingGemma2VisionPatchEmbedder.Weights(
                new float[4][3], positions);
        var attention = new EmbeddingGemma2VisionAttention.Weights(
                zeros, zeros, zeros, zeros,
                new float[]{1f, 1f, 1f, 1f}, new float[]{1f, 1f, 1f, 1f});
        var mlp = new EmbeddingGemma2Mlp.Weights(mlpGate, mlpUp, mlpDown);
        var norms = new EmbeddingGemma2VisionEncoderLayer.Weights(
                new float[]{1f, 1f, 1f, 1f},
                new float[]{1f, 1f, 1f, 1f},
                new float[]{1f, 1f, 1f, 1f},
                new float[]{1f, 1f, 1f, 1f});
        return new EmbeddingGemma2VisionCheckpointMapper.Weights(
                patch,
                List.of(new EmbeddingGemma2VisionCheckpointMapper.LayerWeights(attention, mlp, norms)),
                null,
                standardize ? new float[]{1f, 1f, 1f, 1f} : null,
                standardize ? new float[]{2f, 2f, 2f, 2f} : null);
    }
}
