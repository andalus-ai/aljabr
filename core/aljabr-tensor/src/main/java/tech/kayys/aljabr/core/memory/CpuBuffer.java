package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * CPU-backed off-heap buffer using FFM Arena.
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class CpuBuffer implements Buffer {

    private final Arena arena;
    private final MemorySegment segment;
    private final boolean owned;
    private final OffHeapBufferPool pool;
    private int refCount = 1;

    public CpuBuffer(long sizeBytes) {
        this.arena = Arena.ofShared();
        this.segment = arena.allocate(sizeBytes);
        this.owned = true;
        this.pool = null;
    }

    public CpuBuffer(MemorySegment segment, Arena arena) {
        this(segment, arena, null);
    }

    /** Wraps a segment leased from a pool; the final release returns it to that pool. */
    public CpuBuffer(MemorySegment segment, Arena arena, OffHeapBufferPool pool) {
        this.segment = java.util.Objects.requireNonNull(segment, "segment must not be null");
        this.arena = arena;
        this.pool = pool;
        this.owned = false;
    }

    @Override
    public Arena arena() {
        return arena;
    }

    @Override
    public MemorySegment segment() {
        return segment;
    }

    @Override
    public long sizeBytes() {
        return segment.byteSize();
    }

    @Override
    public synchronized void retain() {
        if (refCount <= 0) {
            throw new IllegalStateException("Cannot retain a released buffer");
        }
        refCount++;
    }

    @Override
    public synchronized void release() {
        if (refCount <= 0) {
            throw new IllegalStateException("Buffer has already been released");
        }
        if (--refCount == 0) {
            if (pool != null) {
                pool.release(segment);
            } else if (owned) {
                arena.close();
            }
        }
    }
}
