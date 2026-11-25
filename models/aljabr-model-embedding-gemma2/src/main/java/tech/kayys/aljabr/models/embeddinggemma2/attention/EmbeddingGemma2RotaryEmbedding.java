package tech.kayys.aljabr.models.embeddinggemma2.attention;

import tech.kayys.aljabr.models.common.transformer.RotaryPositionEmbeddingReference;

/**
 * Compatibility facade for the shared float32 reference RoPE operation.
 *
 * <p>Uses the split-half convention from the Hugging Face reference.
 * Query and key are rotated; values are never passed to this operation.</p>
 */
public final class EmbeddingGemma2RotaryEmbedding {
    private EmbeddingGemma2RotaryEmbedding() {
    }

    /**
     * Applies RoPE to [sequence][heads][headDim] using explicit position ids.
     *
     * @param input query or key tensor
     * @param positions one position id per sequence token
     * @param theta RoPE base for this layer type
     * @param partialRotaryFactor fraction of each head dimension to rotate (null means 1)
     * @return a new tensor; input is not mutated
     */
    public static float[][][] apply(float[][][] input, int[] positions,
            double theta, Double partialRotaryFactor) {
        if (positions != null) {
            for (int position : positions) {
                if (position < 0) {
                    throw new IllegalArgumentException("position ids must not be negative");
                }
            }
        }
        return RotaryPositionEmbeddingReference.apply(input, positions, theta, partialRotaryFactor);
    }
}
