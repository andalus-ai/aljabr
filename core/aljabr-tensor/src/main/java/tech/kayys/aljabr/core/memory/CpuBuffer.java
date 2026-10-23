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
    private final Buffer parent;
    private int refCount = 1;

    public CpuBuffer(long sizeBytes) {
        this.arena = Arena.ofShared();
        this.segment = arena.allocate(sizeBytes);
        this.owned = true;
        this.pool = null;
        this.parent = null;
    }

    public CpuBuffer(MemorySegment segment, Arena arena) {
        this(segment, arena, null);
    }

    /** Wraps a segment leased from a pool; the final release returns it to that pool. */
    public CpuBuffer(MemorySegment segment, Arena arena, OffHeapBufferPool pool) {
        this.segment = java.util.Objects.requireNonNull(segment, "segment must not be null");
        this.arena = arena;
        this.pool = pool;
        this.parent = null;
        this.owned = false;
    }

    /** Wraps a zero-copy view whose lifetime is bounded by a parent buffer. */
    public static CpuBuffer view(MemorySegment segment, Buffer parent) {
        return new CpuBuffer(segment, parent, true);
    }

    private CpuBuffer(MemorySegment segment, Buffer parent, boolean view) {
        this.segment = java.util.Objects.requireNonNull(segment, "segment must not be null");
        this.parent = java.util.Objects.requireNonNull(parent, "parent must not be null");
        // Validate the actual segment range against the parent segment, not only
        // its byte length. A same-sized segment from another allocation must not be
        // accepted as a view whose lifetime is tied to this parent.
        java.util.Objects.requireNonNull(parent.segment(), "parent segment must not be null");
        if (!parent.segment().isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException("Parent buffer is not accessible by the current thread");
        }
        if (segment.byteSize() > parent.sizeBytes()
                || !parent.segment().asSlice(0, parent.sizeBytes()).equals(segment)) {
            // MemorySegment.equals compares segment identity and bounds, not arbitrary
            // overlapping memory addresses; use address/range validation below instead.
            throw new IllegalArgumentException("View segment must be contained within parent buffer");
        }
        this.arena = parent.arena();
        this.pool = null;
        this.owned = false;
        parent.retain();
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
            } else if (parent != null) {
                parent.release();
            } else if (owned) {
                arena.close();
            }
        }
    }
}
