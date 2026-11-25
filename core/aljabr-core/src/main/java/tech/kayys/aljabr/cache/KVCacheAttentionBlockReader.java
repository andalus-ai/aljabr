package tech.kayys.aljabr.cache;

/**
 * Optional storage capability for delivering K and V together in contiguous
 * blocks optimized for attention.
 *
 * <p>Implementations should decode each stored block at most once for K and
 * once for V, then invoke the consumer with the available contiguous segment.
 * This is a read capability only; sequence visibility remains the
 * responsibility of {@link KVCacheSession}.</p>
 */
public interface KVCacheAttentionBlockReader {

    /**
     * Reads a sequence range as contiguous K/V blocks.
     *
     * @param layer cache layer
     * @param firstPosition first sequence position
     * @param tokenCount number of positions
     * @param consumer receives each block in sequence order
     */
    void readAttentionBlocks(
            int layer,
            int firstPosition,
            int tokenCount,
            KVCacheAttentionBlockConsumer consumer);
}
