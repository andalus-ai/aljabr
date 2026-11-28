package tech.kayys.aljabr.cache;

/**
 * A borrowed, read-only view of a bounded prefix of a target-owned KV cache.
 *
 * <p>A read view does not own, allocate, or close the underlying cache storage.
 * Its provider defines the synchronization and lifetime guarantees. Callers
 * must close the view when finished. Implementations must reject access after
 * close and must not expose positions beyond {@link #visibleSequenceLength()}.</p>
 *
 * <p>This API deliberately describes logical KV access rather than a Java
 * array, native pointer, or device allocation. Providers may later expose
 * stronger backend-specific borrowing capabilities without changing this
 * format-neutral contract. A read through this interface may copy data.</p>
 */
public interface KVCacheReadView extends AutoCloseable {

    /** Returns the immutable logical geometry of the borrowed cache. */
    KVCacheLayout layout();

    /** Number of committed positions visible to this view. */
    int visibleSequenceLength();

    /** Copies a contiguous range of K vectors into the caller-owned array. */
    void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset);

    /** Copies a contiguous range of V vectors into the caller-owned array. */
    void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset);

    /** Releases the borrow; it never closes the target-owned storage. */
    @Override
    void close();
}
