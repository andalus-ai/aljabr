package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

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
        if (sizeBytes < 0) {
            throw new IllegalArgumentException("sizeBytes must be non-negative");
        }
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
        this.segment = Objects.requireNonNull(segment, "segment must not be null");
        this.arena = Objects.requireNonNull(arena, "arena must not be null");
        this.pool = pool;
        this.parent = null;
        this.owned = false;
    }

    /**
     * Creates a zero-copy view over a checked slice of {@code parent}.
     * The view retains its parent until the view's final release.
     */
    public static CpuBuffer view(Buffer parent, long offsetBytes, long lengthBytes) {
        Objects.requireNonNull(parent, "parent must not be null");
        if (offsetBytes < 0 || lengthBytes < 0) {
            throw new IllegalArgumentException("View offset and length must be non-negative");
        }
        long end;
        try {
            end = Math.addExact(offsetBytes, lengthBytes);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("View range overflows long", e);
        }
        if (end > parent.sizeBytes()) {
            throw new IllegalArgumentException("View range exceeds parent buffer capacity");
        }
        return new CpuBuffer(parent.segment().asSlice(offsetBytes, lengthBytes), parent, true);
    }

    /**
     * Wraps a segment as a zero-copy view of a parent.
     *
     * @deprecated Prefer {@link #view(Buffer, long, long)} so the slice is
     *             derived from the parent and cannot refer to unrelated memory.
     */
    @Deprecated
    public static CpuBuffer view(MemorySegment segment, Buffer parent) {
        return new CpuBuffer(segment, parent, true);
    }

    private CpuBuffer(MemorySegment segment, Buffer parent, boolean view) {
        this.segment = Objects.requireNonNull(segment, "segment must not be null");
        this.parent = Objects.requireNonNull(parent, "parent must not be null");
        this.arena = Objects.requireNonNull(parent.arena(), "parent arena must not be null");
        this.pool = null;
        this.owned = false;

        // Retain first so the parent cannot be released while the range is checked.
        parent.retain();
        boolean valid = false;
        try {
            validateContainedNativeRange(segment, parent.segment());
            valid = true;
        } finally {
            if (!valid) {
                parent.release();
            }
        }
    }

    /**
     * Legacy segment-based views can only be verified for native segments.
     * A size-only check is insufficient: a smaller unrelated segment must not
     * be accepted as a view of this parent.
     */
    private static void validateContainedNativeRange(
            MemorySegment view, MemorySegment parentSegment) {
        final long viewStart;
        final long parentStart;
        try {
            viewStart = view.address();
            parentStart = parentSegment.address();
        } catch (UnsupportedOperationException e) {
            throw new IllegalArgumentException(
                    "Cannot verify view containment for non-native segments; use view(parent, offset, length)",
                    e);
        }

        final long viewEnd;
        final long parentEnd;
        try {
            viewEnd = Math.addExact(viewStart, view.byteSize());
            parentEnd = Math.addExact(parentStart, parentSegment.byteSize());
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Memory segment address range overflows", e);
        }

        if (viewStart < parentStart || viewEnd > parentEnd) {
            throw new IllegalArgumentException("View segment must be fully contained in the parent buffer");
        }
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
        if (refCount == Integer.MAX_VALUE) {
            throw new IllegalStateException("Buffer reference count overflow");
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
