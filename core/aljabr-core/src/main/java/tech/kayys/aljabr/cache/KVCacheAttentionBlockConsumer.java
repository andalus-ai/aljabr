package tech.kayys.aljabr.cache;

/**
 * Receives one contiguous, position-major K/V segment for attention.
 *
 * <p>The arrays are valid only for the duration of the callback and must be
 * treated as read-only. Consumers that retain data must copy them. The segment
 * begins at {@code elementOffset} in both arrays and contains
 * {@code tokenCount * layerStride} floats.</p>
 */
@FunctionalInterface
public interface KVCacheAttentionBlockConsumer {

    void accept(
            int firstPosition,
            int tokenCount,
            float[] keys,
            float[] values,
            int elementOffset);
}
