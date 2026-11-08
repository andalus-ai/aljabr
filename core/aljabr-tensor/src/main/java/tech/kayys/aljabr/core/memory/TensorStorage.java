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
 */
public interface TensorStorage {

    /** Returns the storage capacity in bytes. */
    long sizeBytes();

    /** Retains this storage for an additional owner. */
    void retain();

    /** Releases one ownership reference to this storage. */
    void release();
}
