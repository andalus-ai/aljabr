package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A reference-counted wrapper around {@link java.lang.foreign.Arena} to support
 * Zero-Copy KV cache sharing and cross-process data handoffs.
 * 
 * <p>
 * Inspired by advanced KV cache management techniques (e.g., ReasonCache), this
 * class enables a single physical memory arena to back multiple logical
 * tensors.
 * The underlying native memory is only freed when the reference count drops to
 * zero.
 * @author bhangun
 */
public final class ManagedArena implements AutoCloseable {

    private final Arena arena;
    private final AtomicInteger refCount;

    private ManagedArena(Arena arena) {
        this.arena = Objects.requireNonNull(arena, "arena must not be null");
        this.refCount = new AtomicInteger(1);
    }

    /**
     * Creates a new managed arena backed by a shared Arena.
     */
    public static ManagedArena create() {
        return new ManagedArena(Arena.ofShared());
    }

    /**
     * Creates a new managed arena wrapping an existing Arena.
     * Use with caution: the provided Arena will be closed when the ref count
     * reaches zero.
     */
    public static ManagedArena wrap(Arena arena) {
        return new ManagedArena(arena);
    }

    /**
     * Allocates memory within this arena.
     */
    public MemorySegment allocate(long bytes) {
        return arena.allocate(bytes);
    }

    /**
     * Returns the underlying Java Arena.
     */
    public Arena raw() {
        return arena;
    }

    /**
     * Increments the reference count.
     * 
     * @return this ManagedArena instance for chaining
     */
    public ManagedArena retain() {
        while (true) {
            int current = refCount.get();
            if (current <= 0) {
                throw new IllegalStateException("Cannot retain an already closed arena");
            }
            if (current == Integer.MAX_VALUE) {
                throw new IllegalStateException("ManagedArena reference count overflow");
            }
            if (refCount.compareAndSet(current, current + 1)) {
                return this;
            }
        }
    }

    /**
     * Decrements the reference count. If it drops to zero, the underlying Arena is
     * closed,
     * freeing the native memory.
     */
    @Override
    public void close() {
        while (true) {
            int current = refCount.get();
            if (current <= 0) {
                throw new IllegalStateException("Over-release of ManagedArena detected");
            }
            int next = current - 1;
            if (refCount.compareAndSet(current, next)) {
                if (next == 0) {
                    arena.close();
                }
                return;
            }
        }
    }

    /**
     * Returns whether the managed arena still owns a live underlying scope.
     *
     * <p>This also reflects an externally closed arena supplied through
     * {@link #wrap(Arena)}.</p>
     */
    public boolean isOpen() {
        return refCount.get() > 0 && arena.scope().isAlive();
    }

    /**
     * Returns the current reference count.
     */
    public int refCount() {
        return refCount.get();
    }
}
