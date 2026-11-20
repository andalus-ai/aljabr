package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;

/**
 * Optional Gemma4 vision post-pooling standardization.
 *
 * <p>Applies the checkpoint-provided feature-wise transform
 * {@code (x - stdBias) * stdScale} in float32. Call this after spatial
 * pooling and its sqrt(hiddenSize) scaling, and only when the vision
 * configuration enables standardization. The caller is responsible for
 * casting the result back to the model's working dtype.</p>
 */
public final class EmbeddingGemma2VisionSoftTokenStandardizer {
    private final int hiddenSize;
    private final float[] stdBias;
    private final float[] stdScale;

    public EmbeddingGemma2VisionSoftTokenStandardizer(
            int hiddenSize, float[] stdBias, float[] stdScale) {
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive");
        }
        Objects.requireNonNull(stdBias, "stdBias must not be null");
        Objects.requireNonNull(stdScale, "stdScale must not be null");
        if (stdBias.length != hiddenSize || stdScale.length != hiddenSize) {
            throw new IllegalArgumentException("stdBias and stdScale must match hiddenSize");
        }
        validateFinite(stdBias, "stdBias");
        validateFinite(stdScale, "stdScale");
        this.hiddenSize = hiddenSize;
        this.stdBias = stdBias.clone();
        this.stdScale = stdScale.clone();
    }

    /**
     * Standardizes pooled vision tokens.
     *
     * @param pooledStates float32 pooled features [tokens][hiddenSize]
     * @return standardized float32 features [tokens][hiddenSize]
     */
    public float[][] standardize(float[][] pooledStates) {
        Objects.requireNonNull(pooledStates, "pooledStates must not be null");
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
                    throw new IllegalArgumentException("pooledStates must contain finite values");
                }
                output[token][feature] = (value - stdBias[feature]) * stdScale[feature];
            }
        }
        return output;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    private static void validateFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain finite values");
            }
        }
    }
}
