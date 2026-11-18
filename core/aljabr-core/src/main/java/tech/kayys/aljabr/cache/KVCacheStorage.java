package tech.kayys.aljabr.cache;

/**
 * Format- and backend-neutral storage for transformer key/value tensors.
 *
 * <p>This contract owns tensor storage only. Sequence admission/commit,
 * request/session ownership, eviction, paging, and scheduling belong to
 * higher-level cache managers. Implementations are not required to use heap
 * arrays; they may use direct or device memory.
 *
 * <p>Unless an implementation documents otherwise, callers must externally
 * synchronize concurrent access to the same layer and position.
 */
public interface KVCacheStorage extends AutoCloseable {

    /** Returns the immutable logical geometry used by this storage. */
    KVCacheLayout layout();

    /**
     * Writes one position's complete K and V vectors for a layer.
     *
     * @param layer layer index
     * @param position sequence position
     * @param key source K vector, exactly {@code layout().layerStride(layer)} floats
     * @param keyOffset offset into {@code key}
     * @param value source V vector, exactly {@code layout().layerStride(layer)} floats
     * @param valueOffset offset into {@code value}
     */
    void write(int layer, int position,
            float[] key, int keyOffset,
            float[] value, int valueOffset);

    /**
     * Copies one position's K vector into the destination.
     *
     * @param destination destination array
     * @param destinationOffset destination offset
     */
    void readKey(int layer, int position, float[] destination, int destinationOffset);

    /** Copies one position's V vector into the destination. */
    void readValue(int layer, int position, float[] destination, int destinationOffset);

    /** Clears all positions for one layer. */
    void clearLayer(int layer);

    /** Clears all stored K and V values. */
    void clear();

    /**
     * Clears all positions at or after {@code firstPosition}, across every
     * layer. This supports truncating a sequence after speculative decoding.
     */
    void truncate(int firstPosition);

    /** Releases the storage resources. Subsequent operations must fail. */
    @Override
    void close();
}
