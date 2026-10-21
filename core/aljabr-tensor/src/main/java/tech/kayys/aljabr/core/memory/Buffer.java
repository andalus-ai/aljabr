package tech.kayys.aljabr.core.memory;

import java.lang.foreign.MemorySegment;

/**
 * Represents a contiguous block of memory used as the backing store for a
 * tensor.
 * 
 * <p>
 * Buffers expose their underlying memory via the Java Foreign Function &amp;
 * Memory (FFM)
 * {@link MemorySegment} API, enabling zero-copy sharing and efficient native
 * operations
 * across boundaries.
 * 
 * <p>
 * Buffers are reference-counted to manage off-heap memory lifecycles safely.
 * Implementations
 * must free the underlying memory when the reference count drops to zero after
 * {@link #release()}
 * is called.
 * @author bhangun
 */
public interface Buffer {
    java.lang.foreign.Arena arena();

    MemorySegment segment();

    long sizeBytes();

    void retain();

    void release();
}
