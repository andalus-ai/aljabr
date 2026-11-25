package tech.kayys.aljabr.cache;

/**
 * Receives one contiguous, position-major K/V block for attention.
 *
 * <p>The arrays are valid only for the duration of the callback and must be
 * treated as read-only. Consumers that retain data must copy it. Both arrays
 * contain {@code tokenCount * layerStride} floats beginning at offset zero.</p>
 */
@FunctionalInterface
public interface KVCacheAttentionBlockConsumer {

    void accept(int firstPosition, int tokenCount, float[] keys, float[] values);
}
