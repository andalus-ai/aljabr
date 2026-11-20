package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper;

/**
 * Maps EmbeddingGemma 2 modality-to-text projection tensors.
 *
 * <p>The Hugging Face state-dict key for the image bridge is
 * {@code embed_vision.embedding_projection.weight}. It is a bias-free linear
 * layer with output-major shape [textHiddenSize][multimodalHiddenSize].
 * The pre-projection RMSNorm is unscaled and therefore has no checkpoint
 * tensor. File-format decoding remains the responsibility of the caller.</p>
 */
public final class EmbeddingGemma2MultimodalCheckpointMapper {
    private static final String VISION_PREFIX = "embed_vision.";
    private static final String VISION_PROJECTION = VISION_PREFIX + "embedding_projection.weight";

    private EmbeddingGemma2MultimodalCheckpointMapper() {
    }

    /**
     * Builds the image-to-text soft-token embedder from a full named checkpoint
     * tensor map. Other model namespaces are ignored; unknown image-bridge
     * tensors are rejected so a changed checkpoint contract cannot be silently
     * partially loaded.
     */
    public static EmbeddingGemma2MultimodalEmbedder mapVisionEmbedder(
            int multimodalHiddenSize,
            int textHiddenSize,
            float epsilon,
            Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors) {
        if (multimodalHiddenSize <= 0 || textHiddenSize <= 0) {
            throw new IllegalArgumentException("hidden dimensions must be positive");
        }
        Objects.requireNonNull(tensors, "tensors must not be null");
        var projectionTensor = tensors.get(VISION_PROJECTION);
        if (projectionTensor == null) {
            throw new IllegalArgumentException("missing required multimodal checkpoint tensor: "
                    + VISION_PROJECTION);
        }
        int[] expected = {textHiddenSize, multimodalHiddenSize};
        if (!java.util.Arrays.equals(projectionTensor.shape(), expected)) {
            throw new IllegalArgumentException(VISION_PROJECTION + " expected shape "
                    + java.util.Arrays.toString(expected) + " but was "
                    + java.util.Arrays.toString(projectionTensor.shape()));
        }
        for (String name : tensors.keySet()) {
            if (name != null && name.startsWith(VISION_PREFIX) && !name.equals(VISION_PROJECTION)) {
                throw new IllegalArgumentException("unsupported or unmapped multimodal checkpoint tensor: " + name);
            }
        }

        float[] flat = projectionTensor.values();
        float[][] projection = new float[textHiddenSize][multimodalHiddenSize];
        for (int row = 0; row < textHiddenSize; row++) {
            System.arraycopy(flat, row * multimodalHiddenSize, projection[row], 0, multimodalHiddenSize);
        }
        return new EmbeddingGemma2MultimodalEmbedder(
                multimodalHiddenSize,
                textHiddenSize,
                epsilon,
                new EmbeddingGemma2MultimodalEmbedder.Weights(projection));
    }
}
