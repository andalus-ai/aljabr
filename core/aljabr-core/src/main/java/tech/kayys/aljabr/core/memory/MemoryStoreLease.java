package tech.kayys.aljabr.core.memory;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A scoped borrow of provider-owned memory.
 *
 * <p>The segment is valid only while this lease remains open. Closing the
 * lease releases the provider's borrow and permits operations such as arena
 * compaction or store shutdown to reclaim the underlying allocation.
 */
public final class MemoryStoreLease implements AutoCloseable {

    private final MemorySegment segment;
    private final Runnable releaser;
    private final AtomicBoolean closed = new AtomicBoolean();

    private MemoryStoreLease(MemorySegment segment, Runnable releaser) {
        this.segment = Objects.requireNonNull(segment, "segment");
        this.releaser = Objects.requireNonNull(releaser, "releaser");
    }

    /**
     * Creates a lease. Intended for storage-provider implementations.
     */
    public static MemoryStoreLease create(MemorySegment segment, Runnable releaser) {
        return new MemoryStoreLease(segment, releaser);
    }

    /**
     * Returns the borrowed segment while this lease is open.
     *
     * @throws IllegalStateException if the lease has been closed
     */
    public MemorySegment segment() {
        if (closed.get()) {
            throw new IllegalStateException("Memory store lease is closed");
        }
        return segment;
    }

    public boolean isClosed() {
        return closed.get();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            releaser.run();
        }
    }
}
