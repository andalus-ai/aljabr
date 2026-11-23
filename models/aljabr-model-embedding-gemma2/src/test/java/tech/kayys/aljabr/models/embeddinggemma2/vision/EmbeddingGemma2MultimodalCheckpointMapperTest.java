package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor;

class EmbeddingGemma2MultimodalCheckpointMapperTest {
    @Test
    void mapsHuggingFaceImageProjectionAndComposesEmbedder() {
        var tensors = Map.of(
                "embed_vision.embedding_projection.weight",
                new CheckpointTensor(new int[]{2, 3}, new float[]{1f, 0f, 0f, 0f, 1f, 0f}),
                "vision_tower.unrelated.tensor",
                new CheckpointTensor(new int[]{1}, new float[]{7f}));

        var embedder = EmbeddingGemma2MultimodalCheckpointMapper.mapVisionEmbedder(
                3, 2, 1e-6f, tensors);
        float[][] result = embedder.project(new float[][]{{3f, 4f, 0f}});

        assertEquals(2, result[0].length);
        assertEquals(1.0392305f, result[0][0], 1e-5f);
        assertEquals(1.3856406f, result[0][1], 1e-5f);
    }

    @Test
    void rejectsMissingProjection() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalCheckpointMapper.mapVisionEmbedder(
                        3, 2, 1e-6f, Map.of()));
    }

    @Test
    void rejectsWrongProjectionShape() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalCheckpointMapper.mapVisionEmbedder(
                        3, 2, 1e-6f,
                        Map.of("embed_vision.embedding_projection.weight",
                                new CheckpointTensor(new int[]{3, 2}, new float[6]))));
    }

    @Test
    void rejectsUnknownImageBridgeTensor() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalCheckpointMapper.mapVisionEmbedder(
                        3, 2, 1e-6f,
                        Map.of(
                                "embed_vision.embedding_projection.weight",
                                new CheckpointTensor(new int[]{2, 3}, new float[6]),
                                "embed_vision.unexpected.weight",
                                new CheckpointTensor(new int[]{1}, new float[]{1f}))));
    }
    @Test
    void mapsHuggingFaceAudioProjectionAndComposesEmbedder() {
        var tensors = Map.of(
                "embed_audio.embedding_projection.weight",
                new CheckpointTensor(new int[]{2, 3}, new float[]{1f, 0f, 0f, 0f, 1f, 0f}),
                "audio_tower.unrelated.tensor",
                new CheckpointTensor(new int[]{1}, new float[]{7f}));

        var embedder = EmbeddingGemma2MultimodalCheckpointMapper.mapAudioEmbedder(
                3, 2, 1e-6f, tensors);
        float[][] result = embedder.project(new float[][]{{3f, 4f, 0f}});

        assertEquals(2, result[0].length);
        assertEquals(1.0392305f, result[0][0], 1e-5f);
        assertEquals(1.3856406f, result[0][1], 1e-5f);
    }

    @Test
    void rejectsMissingAudioProjection() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalCheckpointMapper.mapAudioEmbedder(
                        3, 2, 1e-6f, Map.of()));
    }

    @Test
    void rejectsWrongAudioProjectionShape() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalCheckpointMapper.mapAudioEmbedder(
                        3, 2, 1e-6f,
                        Map.of("embed_audio.embedding_projection.weight",
                                new CheckpointTensor(new int[]{3, 2}, new float[6]))));
    }

    @Test
    void rejectsUnknownAudioBridgeTensor() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2MultimodalCheckpointMapper.mapAudioEmbedder(
                        3, 2, 1e-6f,
                        Map.of(
                                "embed_audio.embedding_projection.weight",
                                new CheckpointTensor(new int[]{2, 3}, new float[6]),
                                "embed_audio.unexpected.weight",
                                new CheckpointTensor(new int[]{1}, new float[]{1f}))));
    }

}
