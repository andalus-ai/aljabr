package tech.kayys.aljabr.gguf.javabackend;

/**
 * Resolves the cache layer consumed by a decoder layer.
 *
 * <p>A layer with a key projection owns its KV entries. A layer without a key
 * projection reuses the preceding global-attention owner, or the owner two
 * layers back for sliding-window attention. The owner is clamped to layer zero
 * to preserve the existing behavior for leading shared layers.</p>
 *
 * <p>This is routing only: it does not allocate, write, advance, or synchronize
 * cache storage.</p>
 */
final class GgufKvCacheRouting {
    private GgufKvCacheRouting() {
    }

    static int resolveCacheLayer(int layer, boolean hasOwnKvProjection, boolean slidingWindowAttention) {
        if (layer < 0) {
            throw new IllegalArgumentException("layer must be non-negative");
        }
        if (hasOwnKvProjection) {
            return layer;
        }
        int ownerDistance = slidingWindowAttention ? 2 : 1;
        return Math.max(0, layer - ownerDistance);
    }
}
