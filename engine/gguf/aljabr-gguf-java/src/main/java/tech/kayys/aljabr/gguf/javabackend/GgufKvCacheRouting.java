package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

/**
 * Resolves the cache layer consumed by a decoder layer.
 *
 * <p>A layer with a key projection owns its KV entries. A layer without a key
 * projection reuses the preceding global-attention owner, or the owner two
 * layers back for sliding-window attention. The owner is clamped to layer zero
 * to preserve the existing behavior for leading shared layers. All execution
 * paths, including native-cache synchronization, must use this resolver.</p>
 *
 * <p>This is routing only: it does not allocate, write, advance, or synchronize
 * cache storage.</p>
 */
final class GgufKvCacheRouting {
    private GgufKvCacheRouting() {
    }

    static void validateCacheGeometry(KVCache cache, int ownerLayer, int expectedKvHeads, int expectedHeadDim) {
        if (cache == null) {
            throw new IllegalArgumentException("KV cache must not be null");
        }
        if (expectedKvHeads <= 0 || expectedHeadDim <= 0) {
            throw new IllegalArgumentException("Expected KV heads and head dimension must be positive");
        }
        int actualKvHeads = cache.nKVHeads(ownerLayer);
        int actualHeadDim = cache.headDim(ownerLayer);
        if (actualKvHeads != expectedKvHeads || actualHeadDim != expectedHeadDim) {
            throw new IllegalArgumentException("KV cache owner layer " + ownerLayer
                    + " geometry mismatch: expected " + expectedKvHeads + " KV heads x " + expectedHeadDim
                    + " dimensions, got " + actualKvHeads + " KV heads x " + actualHeadDim + " dimensions");
        }
    }

    static int resolveCacheLayer(int layer, GGUFTensorInfo[] keyWeights, boolean slidingWindowAttention) {
        if (keyWeights == null) {
            throw new IllegalArgumentException("Key projection metadata must not be null");
        }
        if (layer < 0 || layer >= keyWeights.length) {
            throw new IllegalArgumentException("layer is outside key projection metadata");
        }
        if (keyWeights[layer] != null) {
            return layer;
        }

        int ownerDistance = slidingWindowAttention ? 2 : 1;
        int ownerLayer = Math.max(0, layer - ownerDistance);
        if (keyWeights[ownerLayer] == null) {
            throw new IllegalStateException("Shared KV layer " + layer + " resolves to layer " + ownerLayer
                    + ", but that layer has no key projection and cannot own a KV cache");
        }
        return ownerLayer;
    }
}
