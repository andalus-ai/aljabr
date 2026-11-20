package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;

/**
 * Applies Gemma4's optional checkpoint-backed vision soft-token standardization.
 *
 * <p>The reference operation is feature-wise {@code (x - stdBias) * stdScale}.
 * It runs after spatial pooling and its {@code sqrt(hiddenSize)} scaling, in
 * float32, and before the result is converted to the model's working dtype.
 * Padding tokens are removed by the caller before this operation.</p>
 *
 * <p>This operation is deliberately separate from the encoder and pooler:
 * checkpoints with {@code standardize=false} do not construct or invoke it.</p>
 */
public final class EmbeddingGemma2VisionStandardizer {
    private final int hiddenSize;
    private final float[] stdBias;
    private final float[] stdScale;

    public EmbeddingGemma2VisionStandardizer(int hiddenSize, Weights weights) {
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive");
        }
        Objects.requireNonNull(weights, "weights must not be null");
        this.hiddenSize = hiddenSize;
        this.stdBias = copyVector(weights.stdBias(), hiddenSize, "stdBias");
        this.stdScale = copyVector(weights.stdScale(), hiddenSize, "stdScale");
    }

    /**
     * Standardizes pooled vision tokens in float32.
     *
     * @param pooledStates pooled states [softTokens][hiddenSize], after the
     *                     pooler's sqrt(hiddenSize) scaling and with padding
     *                     tokens already removed
     * @return standardized states [softTokens][hiddenSize]
     */
    public float[][] apply(float[][] pooledStates) {
        Objects.requireNonNull(pooledStates, "pooledStates must not be null");
        if (pooledStates.length == 0) {
            throw new IllegalArgumentException("pooledStates must be non-empty");
        }
        float[][] output = new float[pooledStates.length][hiddenSize];
        for (int token = 0; token < pooledStates.length; token++) {
            float[] row = pooledStates[token];
            if (row == null || row.length != hiddenSize) {
                throw new IllegalArgumentException(
                        "pooledStates must be rectangular with hiddenSize columns");
            }
            for (int feature = 0; feature < hiddenSize; feature++) {
                float value = row[feature];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("pooledStates must contain only finite values");
                }
                // Keep this arithmetic in float32, matching the reference path.
                output[token][feature] = (value - stdBias[feature]) * stdScale[feature];
            }
        }
        return output;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    private static float[] copyVector(float[] source, int size, String name) {
        if (source == null || source.length != size) {
            throw new IllegalArgumentException(name + " must have shape [" + size + "]");
        }
        float[] copy = source.clone();
        for (float value : copy) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain only finite values");
            }
        }
        return copy;
    }

    /** Checkpoint vectors in feature order. */
    public record Weights(float[] stdBias, float[] stdScale) {
        public Weights {
            Objects.requireNonNull(stdBias, "stdBias must not be null");
            Objects.requireNonNull(stdScale, "stdScale must not be null");
        }
    }
}
