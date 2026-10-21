package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A reference-counted wrapper around a {@link java.lang.foreign.Arena}.
 *
 * <p>
 * Native inference components that share a {@link MemorySegment} across
 * boundaries (e.g. zero-copy from the KV-cache store into a CUDA kernel) need
 * to guarantee that the backing {@code Arena} stays open for the lifetime of
 * <em>all</em> consumers, not just the original producer. A simple
 * {@code try-with-resources} on the producer side is not safe.
 *
 * <p>
 * Use {@link #retain()} to increment the reference count before sharing the
 * arena across a boundary and {@link #close()} (i.e. release) once the
 * consumer is done. The underlying native memory is freed only when the count
 * reaches zero.
 *
 * <pre>{@code
 * ManagedArena ma = new ManagedArena(Arena.ofShared());
 * MemorySegment segment = ma.allocate(1024);
 *
 * // Hand off to a consumer — bump the count first.
 * ma.retain();
 * cudaKernel.enqueue(segment, ma); // consumer calls ma.close() when done
 *
 * // Producer is done; drops its own reference.
 * ma.close();
 * }</pre>
 * @author bhangun
 */
public final class ManagedArena implements AutoCloseable {

    private final Arena arena;
    private final AtomicInteger refCount = new AtomicInteger(1);

    /** Wraps an existing arena. The initial reference count is 1. */
    public ManagedArena(Arena arena) {
        this.arena = arena;
    }

    /** Creates a new shared (thread-safe) arena with ref-count 1. */
    public static ManagedArena ofShared() {
        return new ManagedArena(Arena.ofShared());
    }

    /** Creates a new confined (single-thread) arena with ref-count 1. */
    public static ManagedArena ofConfined() {
        return new ManagedArena(Arena.ofConfined());
    }

    // ── Memory operations ─────────────────────────────────────────────────────

    /**
     * Allocates {@code byteSize} bytes from the underlying arena.
     *
     * @param byteSize number of bytes to allocate
     * @return a new {@link MemorySegment} backed by this arena
     * @throws IllegalStateException if the arena has already been closed
     */
    public MemorySegment allocate(long byteSize) {
        return arena.allocate(byteSize);
    }

    /** Returns the underlying {@link Arena}. */
    public Arena arena() {
        return arena;
    }

    // ── Reference counting ────────────────────────────────────────────────────

    /**
     * Increments the reference count by one and returns {@code this} for
     * chaining.
     *
     * <p>
     * Call this immediately before passing this arena to a new consumer.
     *
     * @return {@code this}
     */
    public ManagedArena retain() {
        int prev = refCount.getAndIncrement();
        if (prev <= 0) {
            // Reverse the increment — the arena is already closed.
            refCount.decrementAndGet();
            throw new IllegalStateException("Cannot retain a closed ManagedArena");
        }
        return this;
    }

    /**
     * Decrements the reference count. When the count reaches zero the
     * underlying arena — and all memory allocated from it — is freed.
     */
    @Override
    public void close() {
        if (refCount.decrementAndGet() == 0) {
            arena.close();
        }
    }

    /** Returns the current reference count (for diagnostics / tests). */
    public int refCount() {
        return refCount.get();
    }

    /** {@code true} if the underlying arena is still open. */
    public boolean isOpen() {
        return refCount.get() > 0;
    }
}
