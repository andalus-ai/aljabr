package tech.kayys.aljabr.core.memory;

/**
 * Backend-neutral storage owned by a tensor.
 *
 * <p>This contract describes storage capacity and lifetime only. It does not
 * assume that storage is backed by a Java heap array, a host address, a
 * {@link java.lang.foreign.MemorySegment}, or device-local memory. The tensor
 * carries its logical device metadata separately.</p>
 *
 * <p>Each successful {@link #retain()} adds one ownership reference. Every
 * owner must eventually balance that reference with {@link #release()}.
 * Implementations define their behavior after the final release; callers
 * must not access storage once they have relinquished their last reference.</p>
 *
 * <p>Methods returning a {@code TensorStorage} from a tensor generally expose a
 * borrowed reference. Such a reference does not represent an additional owner.
 * Call {@link #retain()} to establish an independent ownership reference and
 * balance it with exactly one {@link #release()}.</p>
 */
public interface TensorStorage {

    /** Returns the storage capacity in bytes. */
    long sizeBytes();

    /**
     * Acquires one additional ownership reference.
     *
     * @throws IllegalStateException if the storage has already been released
     *         or cannot accept another reference
     */
    void retain();

    /**
     * Relinquishes one ownership reference. The final release ends the
     * storage lifetime; using the storage afterward is invalid.
     *
     * @throws IllegalStateException if no live reference remains
     */
    void release();
}
