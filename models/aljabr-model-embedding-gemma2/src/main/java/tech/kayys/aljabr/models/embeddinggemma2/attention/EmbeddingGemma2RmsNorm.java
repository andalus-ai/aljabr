package tech.kayys.aljabr.models.embeddinggemma2.attention;

import tech.kayys.aljabr.models.common.transformer.RmsNormReference;

/**
 * Compatibility facade for the shared float32 transformer RMSNorm reference.
 *
 * <p>The scale is optional: query/key norms have learned scales, while the
 * value norm is unscaled.</p>
 */
public final class EmbeddingGemma2RmsNorm {
    private EmbeddingGemma2RmsNorm() {
    }

    /** Normalizes [sequence][heads][headDim], returning a new tensor. */
    public static float[][][] apply(float[][][] input, float[] scale, float epsilon) {
        return RmsNormReference.apply(input, scale, epsilon);
    }
}
