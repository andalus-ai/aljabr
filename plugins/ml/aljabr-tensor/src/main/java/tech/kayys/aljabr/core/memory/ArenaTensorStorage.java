package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Reference-counted host storage backed by a dedicated shared FFM arena.
 *
 * <p>The storage owns its arena. The arena is closed when the final storage
 * reference is released. A tensor or view must retain the storage when it
 * acquires an independent ownership reference and release it exactly once
 * when that ownership ends.</p>
 */
public final class ArenaTensorStorage implements HostAccessibleStorage {

    private final Arena arena;
    private final MemorySegment segment;
    private final AtomicInteger references = new AtomicInteger(1);

    ArenaTensorStorage(Arena arena, MemorySegment segment) {
        this.arena = Objects.requireNonNull(arena, "arena must not be null");
        this.segment = Objects.requireNonNull(segment, "segment must not be null");
    }

    @Override
    public MemorySegment segment() {
        ensureOpen();
        return segment;
    }

    @Override
    public long sizeBytes() {
        ensureOpen();
        return segment.byteSize();
    }

    @Override
    public void retain() {
        while (true) {
            int current = references.get();
            if (current <= 0) {
                throw new IllegalStateException("Tensor storage has already been released");
            }
            if (current == Integer.MAX_VALUE) {
                throw new IllegalStateException("Tensor storage reference count overflow");
            }
            if (references.compareAndSet(current, current + 1)) {
                return;
            }
        }
    }

    @Override
    public void release() {
        while (true) {
            int current = references.get();
            if (current <= 0) {
                throw new IllegalStateException("Tensor storage has already been released");
            }
            int next = current - 1;
            if (references.compareAndSet(current, next)) {
                if (next == 0) {
                    arena.close();
                }
                return;
            }
        }
    }

    /** Returns the current number of outstanding ownership references. */
    public int referenceCount() {
        return references.get();
    }

    private void ensureOpen() {
        if (references.get() <= 0) {
            throw new IllegalStateException("Tensor storage has already been released");
        }
    }
}
