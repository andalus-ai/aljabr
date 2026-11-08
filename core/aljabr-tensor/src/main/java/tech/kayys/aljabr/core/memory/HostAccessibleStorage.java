package tech.kayys.aljabr.core.memory;

import java.lang.foreign.MemorySegment;

/**
 * Optional capability for tensor storage that can be accessed from the host.
 *
 * <p>A host-accessible storage may be native, mapped, or backed by another
 * memory segment implementation. The returned segment is valid only while
 * the caller holds a live ownership reference to this storage.</p>
 *
 * <p>Device-local storage that cannot be represented as a host-accessible
 * {@link MemorySegment} should implement {@link TensorStorage} directly
 * instead of fabricating a host segment.</p>
 */
public interface HostAccessibleStorage extends TensorStorage {

    /** Returns a host-accessible view of this storage. */
    MemorySegment segment();
}
