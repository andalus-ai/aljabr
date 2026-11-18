package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Validates the output projection that maps concatenated query-head context
 * back into the model embedding space.
 */
final class GgufAttentionOutputProjectionLayout {
    private GgufAttentionOutputProjectionLayout() {
    }

    static void validate(
            GGUFTensorInfo[] outputWeights,
            int queryHeads,
            int[] headDimensions,
            int embeddingDimension) {
        if (outputWeights == null || headDimensions == null) {
            throw new IllegalArgumentException("attention output projection layout arrays must not be null");
        }
        if (embeddingDimension <= 0 || queryHeads <= 0) {
            throw new IllegalArgumentException("attention output projection dimensions must be positive");
        }
        if (outputWeights.length != headDimensions.length) {
            throw new IllegalArgumentException(
                    "attention output projection and head-dimension arrays must have equal lengths");
        }

        for (int layer = 0; layer < outputWeights.length; layer++) {
            GGUFTensorInfo tensor = outputWeights[layer];
            if (tensor == null) {
                throw new IllegalArgumentException(
                        "Layer " + layer + " is missing its attention output projection");
            }
            int headDimension = headDimensions[layer];
            if (headDimension <= 0) {
                throw new IllegalArgumentException(
                        "query head dimension must be positive for layer " + layer);
            }

            long expectedInputDimension = Math.multiplyExact((long) queryHeads, headDimension);
            long actualInputDimension = GgufTensorOps.matrixColumns(tensor);
            long actualOutputDimension = GgufTensorOps.matrixRows(tensor);
            if (actualInputDimension != expectedInputDimension
                    || actualOutputDimension != embeddingDimension) {
                throw new IllegalArgumentException("Layer " + layer
                        + " attention output projection tensor '" + tensor.name()
                        + "' has matrix shape [" + actualOutputDimension + ", "
                        + actualInputDimension + "]; expected [" + embeddingDimension
                        + ", " + expectedInputDimension + "] from query-head geometry");
            }
        }
    }
}
