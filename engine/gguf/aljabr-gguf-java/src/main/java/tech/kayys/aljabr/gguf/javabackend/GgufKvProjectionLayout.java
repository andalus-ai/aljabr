package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Validates the Java GGUF runner's per-layer key/value projection topology
 * and matrix geometry before a decoder session is initialized.
 */
final class GgufKvProjectionLayout {
    private GgufKvProjectionLayout() {
    }

    static void validate(
            GGUFTensorInfo[] keyWeights,
            GGUFTensorInfo[] valueWeights,
            int[] kvHeads,
            int[] headDimensions,
            boolean[] slidingWindowLayers,
            int embeddingDimension) {
        if (keyWeights == null || valueWeights == null || kvHeads == null
                || headDimensions == null || slidingWindowLayers == null) {
            throw new IllegalArgumentException("KV projection layout arrays must not be null");
        }
        if (embeddingDimension <= 0) {
            throw new IllegalArgumentException("embedding dimension must be positive");
        }

        int layers = keyWeights.length;
        if (valueWeights.length != layers || kvHeads.length != layers
                || headDimensions.length != layers || slidingWindowLayers.length != layers) {
            throw new IllegalArgumentException("KV projection layout arrays must have equal lengths");
        }

        for (int layer = 0; layer < layers; layer++) {
            if (kvHeads[layer] <= 0 || headDimensions[layer] <= 0) {
                throw new IllegalArgumentException("KV geometry must be positive for layer " + layer);
            }
            boolean hasKey = keyWeights[layer] != null;
            boolean hasValue = valueWeights[layer] != null;
            if (!hasKey && hasValue) {
                throw new IllegalArgumentException("Layer " + layer
                        + " has a value projection but no key projection");
            }

            if (hasKey) {
                validateProjection(keyWeights[layer], layer, "key", embeddingDimension,
                        kvHeads[layer], headDimensions[layer]);
            }
            if (hasValue) {
                validateProjection(valueWeights[layer], layer, "value", embeddingDimension,
                        kvHeads[layer], headDimensions[layer]);
            }

            if (!hasKey) {
                int owner = GgufKvCacheRouting.resolveCacheLayer(
                        layer, keyWeights, slidingWindowLayers[layer]);
                if (kvHeads[layer] != kvHeads[owner]
                        || headDimensions[layer] != headDimensions[owner]) {
                    throw new IllegalArgumentException("Shared KV layer " + layer
                            + " has geometry (" + kvHeads[layer] + " KV heads, "
                            + headDimensions[layer] + " head dimension) incompatible with owner layer "
                            + owner + " (" + kvHeads[owner] + " KV heads, "
                            + headDimensions[owner] + " head dimension)");
                }
            }
        }
    }

    private static void validateProjection(
            GGUFTensorInfo tensor, int layer, String projection,
            int expectedInputDimension, int kvHeads, int headDimension) {
        long expectedOutputDimension = Math.multiplyExact((long) kvHeads, headDimension);
        long actualInputDimension = GgufTensorOps.matrixColumns(tensor);
        long actualOutputDimension = GgufTensorOps.matrixRows(tensor);
        if (actualInputDimension != expectedInputDimension
                || actualOutputDimension != expectedOutputDimension) {
            throw new IllegalArgumentException("Layer " + layer + " " + projection
                    + " projection tensor '" + tensor.name() + "' has matrix shape ["
                    + actualOutputDimension + ", " + actualInputDimension + "]"
                    + "; expected [" + expectedOutputDimension + ", "
                    + expectedInputDimension + "] from KV geometry");
        }
    }
}
