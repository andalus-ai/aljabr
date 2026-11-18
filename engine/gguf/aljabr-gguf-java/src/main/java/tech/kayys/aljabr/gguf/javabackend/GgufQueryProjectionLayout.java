package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Validates query projection matrices against the model's query-head geometry.
 */
final class GgufQueryProjectionLayout {
    private GgufQueryProjectionLayout() {
    }

    static void validate(
            GGUFTensorInfo[] queryWeights,
            int queryHeads,
            int[] headDimensions,
            int embeddingDimension) {
        if (queryWeights == null || headDimensions == null) {
            throw new IllegalArgumentException("query projection layout arrays must not be null");
        }
        if (embeddingDimension <= 0 || queryHeads <= 0) {
            throw new IllegalArgumentException("query projection dimensions must be positive");
        }
        if (queryWeights.length != headDimensions.length) {
            throw new IllegalArgumentException(
                    "query projection and head-dimension arrays must have equal lengths");
        }

        for (int layer = 0; layer < queryWeights.length; layer++) {
            GGUFTensorInfo tensor = queryWeights[layer];
            if (tensor == null) {
                throw new IllegalArgumentException("Layer " + layer + " is missing its query projection");
            }
            int headDimension = headDimensions[layer];
            if (headDimension <= 0) {
                throw new IllegalArgumentException("query head dimension must be positive for layer " + layer);
            }

            long expectedOutputDimension = Math.multiplyExact((long) queryHeads, headDimension);
            long actualInputDimension = GgufTensorOps.matrixColumns(tensor);
            long actualOutputDimension = GgufTensorOps.matrixRows(tensor);
            if (actualInputDimension != embeddingDimension
                    || actualOutputDimension != expectedOutputDimension) {
                throw new IllegalArgumentException("Layer " + layer
                        + " query projection tensor '" + tensor.name()
                        + "' has matrix shape [" + actualOutputDimension + ", "
                        + actualInputDimension + "]; expected [" + expectedOutputDimension
                        + ", " + embeddingDimension + "] from query-head geometry");
            }
        }
    }
}
