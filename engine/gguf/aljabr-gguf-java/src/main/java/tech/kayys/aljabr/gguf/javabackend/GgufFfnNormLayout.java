package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

/**
 * Validates the required FFN input norm and optional post-FFN norm vectors
 * before they are materialized for GGUF Java execution.
 */
final class GgufFfnNormLayout {
    private GgufFfnNormLayout() {
    }

    static void validateRequiredTensor(
            GGUFTensorInfo tensor, String tensorName, int layer, int expectedDimension) {
        if (tensor == null) {
            throw new IllegalArgumentException("GGUF required FFN norm tensor '" + tensorName
                    + "' is missing for layer " + layer);
        }
        validatePresentTensor(tensor, tensorName, layer, expectedDimension);
    }

    /**
     * A post-FFN norm is architecture-specific and may be omitted. When present,
     * it must be a rank-1 vector matching the model embedding dimension.
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
            throw new IllegalArgumentException("FFN norm embedding dimension for layer " + layer
                    + " must be positive, got " + expectedDimension);
        }
        long[] shape = tensor.shape();
        if (shape == null || shape.length != 1) {
            throw new IllegalArgumentException("GGUF FFN norm tensor '" + tensorName
                    + "' for layer " + layer + " must be a rank-1 vector");
        }
        if (shape[0] != expectedDimension) {
            throw new IllegalArgumentException("GGUF FFN norm tensor '" + tensorName
                    + "' for layer " + layer + " must contain " + expectedDimension
                    + " values, got " + shape[0]);
        }
    }
}
