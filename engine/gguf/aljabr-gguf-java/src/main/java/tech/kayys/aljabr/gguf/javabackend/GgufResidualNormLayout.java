package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

/**
 * Validates residual-stream normalization vectors before GGUF Java execution
 * materializes them.
 */
final class GgufResidualNormLayout {
    private GgufResidualNormLayout() {
    }

    static void validateRequiredTensor(
            GGUFTensorInfo tensor, String tensorName, int layer, int expectedDimension) {
        if (tensor == null) {
            throw new IllegalArgumentException("GGUF required residual norm tensor '" + tensorName
                    + "' is missing for layer " + layer);
        }
        validatePresentTensor(tensor, tensorName, layer, expectedDimension);
    }

    /**
     * Post-attention and per-layer post norms are architecture-specific.
     * Missing tensors are valid; present tensors must match the residual width.
     */
    static void validateOptionalTensor(
            GGUFTensorInfo tensor, String tensorName, int layer, int expectedDimension) {
        if (tensor != null) {
            validatePresentTensor(tensor, tensorName, layer, expectedDimension);
        }
    }

    private static void validatePresentTensor(
            GGUFTensorInfo tensor, String tensorName, int layer, int expectedDimension) {
        if (expectedDimension <= 0) {
            throw new IllegalArgumentException("residual norm embedding dimension for layer " + layer
                    + " must be positive, got " + expectedDimension);
        }
        long[] shape = tensor.shape();
        if (shape == null || shape.length != 1) {
            throw new IllegalArgumentException("GGUF residual norm tensor '" + tensorName
                    + "' for layer " + layer + " must be a rank-1 vector");
        }
        if (shape[0] != expectedDimension) {
            throw new IllegalArgumentException("GGUF residual norm tensor '" + tensorName
                    + "' for layer " + layer + " must contain " + expectedDimension
                    + " values, got " + shape[0]);
        }
    }
}
