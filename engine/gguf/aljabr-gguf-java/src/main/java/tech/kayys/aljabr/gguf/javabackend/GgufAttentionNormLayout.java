package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

/**
 * Validates optional per-head Q/K/V normalization vectors before they are
 * materialized and consumed by the GGUF Java attention path.
 */
final class GgufAttentionNormLayout {
    private GgufAttentionNormLayout() {
    }

    /**
     * Validates a present normalization vector. Missing vectors are valid:
     * architectures may omit Q/K norms, and V may use unit normalization
     * without a learned scale.
     */
    static void validateTensor(
            GGUFTensorInfo tensor, String tensorName, int layer, int expectedDimension) {
        if (expectedDimension <= 0) {
            throw new IllegalArgumentException("attention head dimension for layer " + layer
                    + " must be positive, got " + expectedDimension);
        }
        if (tensor == null) {
            return;
        }

        long[] shape = tensor.shape();
        if (shape == null || shape.length != 1) {
            throw new IllegalArgumentException("GGUF attention norm tensor '" + tensorName
                    + "' for layer " + layer + " must be a rank-1 vector");
        }
        if (shape[0] != expectedDimension) {
            throw new IllegalArgumentException("GGUF attention norm tensor '" + tensorName
                    + "' for layer " + layer + " must contain " + expectedDimension
                    + " values, got " + shape[0]);
        }
    }
}
