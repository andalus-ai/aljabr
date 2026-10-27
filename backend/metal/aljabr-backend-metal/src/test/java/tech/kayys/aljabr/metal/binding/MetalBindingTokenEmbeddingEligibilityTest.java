package tech.kayys.aljabr.metal.binding;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalBindingTokenEmbeddingEligibilityTest {
    private static final int GGML_TYPE_Q4_K = 12;
    private static final int GGML_TYPE_Q5_K = 13;

    @Test
    void acceptsQ4KWithValidShapeAndFiniteScale() {
        assertTrue(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, 3840, 1.0f));
        assertTrue(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, 256, 1.0f));
    }

    @Test
    void rejectsUnsupportedQuantizationLayouts() {
        assertFalse(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q5_K, 3840, 1.0f));
        assertFalse(MetalBinding.isTokenEmbeddingEligible(0, 3840, 1.0f));
    }

    @Test
    void rejectsInvalidDimensionsAndNonFiniteScale() {
        assertFalse(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, 0, 1.0f));
        assertFalse(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, -1, 1.0f));
        assertFalse(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, 255, 1.0f));
        assertFalse(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, 257, 1.0f));
        assertFalse(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, 3840, Float.NaN));
        assertFalse(MetalBinding.isTokenEmbeddingEligible(GGML_TYPE_Q4_K, 3840, Float.POSITIVE_INFINITY));
    }
}
