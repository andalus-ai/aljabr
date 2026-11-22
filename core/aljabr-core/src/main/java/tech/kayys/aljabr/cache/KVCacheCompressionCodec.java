package tech.kayys.aljabr.cache;

/**
 * Encodes and decodes contiguous transformer KV-cache blocks.
 *
 * <p>This is a numerical codec contract, not a storage manager or serving
 * policy. Implementations must not own request admission, sequence commit,
 * eviction, paging, or scheduling. They must not retain caller-owned input
 * arrays or rely on mutable process-global state to decode a block.</p>
 *
 * <p>The encoded block must carry all per-block decoder metadata. Any reusable
 * immutable algorithm state may be owned by the codec instance, but the codec
 * identity and version must remain stable for the lifetime of encoded blocks
 * produced by that instance.</p>
 */
public interface KVCacheCompressionCodec {

    /** Stable, format-neutral identifier for this codec implementation. */
    String id();

    /** Positive version of the encoded block representation. */
    int version();

    /**
     * Encodes exactly {@code context.elementCount()} floats from the source.
     *
     * @param source source float array
     * @param sourceOffset first source element
     * @param context logical geometry and location of the block
     * @return immutable encoded payload and its decoder metadata
     */
    KVCacheEncodedBlock encode(
            float[] source,
            int sourceOffset,
            KVCacheCompressionContext context);

    /**
     * Decodes a block into exactly {@code encoded.elementCount()} destination
     * elements beginning at {@code destinationOffset}.
     *
     * <p>Implementations must reject blocks with an unsupported codec identity,
     * version, or geometry rather than silently interpreting incompatible data.</p>
     *
     * @param encoded encoded payload and decoder metadata
     * @param destination destination float array
     * @param destinationOffset first destination element
     * @param context logical geometry and location used during encoding
     */
    void decode(
            KVCacheEncodedBlock encoded,
            float[] destination,
            int destinationOffset,
            KVCacheCompressionContext context);
}
