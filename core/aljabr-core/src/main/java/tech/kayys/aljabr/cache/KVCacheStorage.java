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

    /**
     * Writes a contiguous sequence range for one layer. The source arrays
     * contain {@code tokenCount * layout().layerStride(layer)} floats beginning
     * at their respective offsets, in position-major order.
     *
     * <p>The default implementation delegates to {@link #write}; optimized
     * providers should override it with a bulk transfer where possible.
     */
    default void writeRange(int layer, int firstPosition, int tokenCount,
            float[] keys, int keysOffset, float[] values, int valuesOffset) {
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(keys, keysOffset, elements, "keys");
        checkArrayRange(values, valuesOffset, elements, "values");
        for (int token = 0; token < tokenCount; token++) {
            int sourceOffset = token * stride;
            write(layer, firstPosition + token,
                    keys, keysOffset + sourceOffset,
                    values, valuesOffset + sourceOffset);
        }
    }

    /** Copies a contiguous sequence range of K vectors into {@code destination}. */
    default void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        for (int token = 0; token < tokenCount; token++) {
            readKey(layer, firstPosition + token, destination,
                    destinationOffset + token * stride);
        }
    }

    /** Copies a contiguous sequence range of V vectors into {@code destination}. */
    default void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        for (int token = 0; token < tokenCount; token++) {
            readValue(layer, firstPosition + token, destination,
                    destinationOffset + token * stride);
        }
    }

    private int checkedRange(int layer, int firstPosition, int tokenCount) {
        KVCacheLayout geometry = layout();
        int stride = geometry.layerStride(layer);
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > geometry.maxSequenceLength()
                || tokenCount > geometry.maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV cache range exceeds layer capacity");
        }
        return stride;
    }

    private static void checkArrayRange(
            float[] array, int offset, int length, String name) {
        if (array == null || offset < 0 || length < 0 || offset > array.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the supplied array");
        }
    }

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
