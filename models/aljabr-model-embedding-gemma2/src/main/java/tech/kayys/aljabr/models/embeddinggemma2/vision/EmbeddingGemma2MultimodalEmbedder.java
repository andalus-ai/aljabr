package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2LinearProjection;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2SequenceRmsNorm;

/**
 * Projects modality-tower soft tokens into the EmbeddingGemma 2 text hidden
 * space. This is the checkpoint-backed bridge between a modality tower and
 * the shared bidirectional text encoder; it is not a pooled-vector adapter.
 *
 * <p>The reference operation is RMSNorm without a learned scale, followed by
 * a bias-free linear projection. The projection weights use output-major
 * {@code [textHiddenSize][multimodalHiddenSize]} layout.</p>
 *
 * <p>The vision tower itself and image preprocessing are deliberately separate
 * components. This class consumes the vision tower's per-patch hidden states
 * and preserves each image's soft-token sequence.</p>
 */
public final class EmbeddingGemma2MultimodalEmbedder {
    private final int multimodalHiddenSize;
    private final int textHiddenSize;
    private final float epsilon;
    private final float[][] projection;

    public EmbeddingGemma2MultimodalEmbedder(
            int multimodalHiddenSize,
            int textHiddenSize,
            float epsilon,
            Weights weights) {
        if (multimodalHiddenSize <= 0 || textHiddenSize <= 0) {
            throw new IllegalArgumentException("hidden dimensions must be positive");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        Objects.requireNonNull(weights, "weights must not be null");
        this.multimodalHiddenSize = multimodalHiddenSize;
        this.textHiddenSize = textHiddenSize;
        this.epsilon = epsilon;
        this.projection = copyAndValidate(weights.projection(), textHiddenSize, multimodalHiddenSize);
    }

    /** Projects one image's soft-token sequence [tokens][multimodalHiddenSize]. */
    public float[][] project(float[][] modalityStates) {
        validateStates(modalityStates);
        float[][] normalized = EmbeddingGemma2SequenceRmsNorm.apply(
                modalityStates, null, epsilon);
        return EmbeddingGemma2LinearProjection.apply(normalized, projection, null);
    }

    /**
     * Projects a collection of images independently, retaining image boundaries
     * and token order. Each result has shape [softTokens][textHiddenSize].
     */
    public List<float[][]> projectImages(List<float[][]> imageStates) {
        Objects.requireNonNull(imageStates, "imageStates must not be null");
        if (imageStates.isEmpty()) {
            throw new IllegalArgumentException("at least one image sequence is required");
        }
        List<float[][]> projected = new ArrayList<>(imageStates.size());
        for (int image = 0; image < imageStates.size(); image++) {
            try {
                projected.add(project(imageStates.get(image)));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("invalid vision states for image " + image + ": "
                        + e.getMessage(), e);
            }
        }
        return List.copyOf(projected);
    }

    public int multimodalHiddenSize() {
        return multimodalHiddenSize;
    }

    public int textHiddenSize() {
        return textHiddenSize;
    }

    private void validateStates(float[][] states) {
        if (states == null || states.length == 0) {
            throw new IllegalArgumentException("modality states must contain at least one soft token");
        }
        for (int token = 0; token < states.length; token++) {
            if (states[token] == null || states[token].length != multimodalHiddenSize) {
                throw new IllegalArgumentException("modality states must have shape [tokens]["
                        + multimodalHiddenSize + "]; invalid token " + token);
            }
        }
    }

    private static float[][] copyAndValidate(float[][] source, int rows, int columns) {
        if (source == null || source.length != rows) {
            throw new IllegalArgumentException("projection must have shape [" + rows + "][" + columns + "]");
        }
        float[][] copy = new float[rows][columns];
        for (int row = 0; row < rows; row++) {
            if (source[row] == null || source[row].length != columns) {
                throw new IllegalArgumentException("projection must have shape [" + rows + "][" + columns + "]");
            }
            for (int column = 0; column < columns; column++) {
                float value = source[row][column];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("projection must contain finite values");
                }
                copy[row][column] = value;
            }
        }
        return copy;
    }

    /** Checkpoint tensor for the bias-free modality-to-text projection. */
    public record Weights(float[][] projection) {
        public Weights {
            Objects.requireNonNull(projection, "projection must not be null");
        }
    }
}
