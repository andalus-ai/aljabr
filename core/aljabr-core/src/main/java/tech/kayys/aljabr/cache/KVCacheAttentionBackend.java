package tech.kayys.aljabr.cache;

/**
 * Optional backend capability for attention directly over a session-backed KV cache.
 *
 * <p>This capability is separate from {@code ComputeBackend}'s dense tensor
 * attention operations. A backend may implement it when it can consume cache
 * blocks without materializing dense key/value tensors. Callers should discover
 * it with {@code instanceof} and retain their existing dense-attention fallback
 * when it is unavailable.</p>
 *
 * <p>The caller retains ownership of the session and its storage. Implementations
 * must respect the session's committed-prefix visibility rules and must not
 * close or otherwise take ownership of the supplied session.</p>
 */
public interface KVCacheAttentionBackend {

    /**
     * Computes one query head against a visible range of the KV cache.
     *
     * @param session sequence-scoped cache session owned by the caller
     * @param layer cache layer
     * @param query one query vector matching the cache head dimension
     * @param queryHeadIndex index of this head among query heads
     * @param queryHeadCount total query heads for the layer
     * @param queryPosition absolute query position for causal masking
     * @param firstKeyPosition first absolute key position to read
     * @param tokenCount number of key positions to read
     * @param causal whether keys after the query position are excluded
     * @return newly allocated attention output vector
     */
    float[] attentionFromCache(
            KVCacheSession session,
            int layer,
            float[] query,
            int queryHeadIndex,
            int queryHeadCount,
            int queryPosition,
            int firstKeyPosition,
            int tokenCount,
            boolean causal);
}
