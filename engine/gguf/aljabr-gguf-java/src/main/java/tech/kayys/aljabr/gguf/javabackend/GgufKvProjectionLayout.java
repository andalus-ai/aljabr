package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

/**
 * Validates the Java GGUF runner's per-layer key/value projection topology
 * before a decoder session is initialized.
 */
final class GgufKvProjectionLayout {
    private GgufKvProjectionLayout() {
    }

    static void validate(
            GGUFTensorInfo[] keyWeights,
            GGUFTensorInfo[] valueWeights,
            int[] kvHeads,
            int[] headDimensions,
            boolean[] slidingWindowLayers) {
        if (keyWeights == null || valueWeights == null || kvHeads == null
                || headDimensions == null || slidingWindowLayers == null) {
            throw new IllegalArgumentException("KV projection layout arrays must not be null");
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

            // K-only is an intentional architecture variant: the runner uses K as V.
            // V-only cannot be represented by the current projection/cache contract.
            if (!hasKey && hasValue) {
                throw new IllegalArgumentException("Layer " + layer
                        + " has a value projection but no key projection");
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
}
