package tech.kayys.aljabr.core.memory;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Reference-counted zero-copy view over a range of host-accessible tensor
 * storage. The view owns one reference to its parent for its entire lifetime.
 */
public final class HostAccessibleStorageView implements HostAccessibleStorage {

    private final HostAccessibleStorage parent;
    private final MemorySegment segment;
    private int references = 1;

    private HostAccessibleStorageView(HostAccessibleStorage parent, MemorySegment segment) {
        this.parent = parent;
        this.segment = segment;
    }

    /**
     * Creates a checked zero-copy view and retains {@code parent}.
     *
     * @param parent host-accessible parent storage
     * @param offsetBytes start offset in bytes
     * @param lengthBytes view length in bytes
     * @return a new independently reference-counted storage view
     */
    public static HostAccessibleStorageView view(
            HostAccessibleStorage parent, long offsetBytes, long lengthBytes) {
        Objects.requireNonNull(parent, "parent must not be null");
        if (offsetBytes < 0 || lengthBytes < 0) {
            throw new IllegalArgumentException("View offset and length must be non-negative");
        }

        final long end;
        try {
            end = Math.addExact(offsetBytes, lengthBytes);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("View range overflows long", e);
        }
        if (end > parent.sizeBytes()) {
            throw new IllegalArgumentException("View range exceeds parent storage capacity");
        }

        parent.retain();
        try {
            return new HostAccessibleStorageView(
                    parent, parent.segment().asSlice(offsetBytes, lengthBytes));
        } catch (RuntimeException | Error e) {
            parent.release();
            throw e;
        }
    }

    @Override
    public synchronized MemorySegment segment() {
        ensureOpen();
        return segment;
    }

    @Override
    public synchronized long sizeBytes() {
        ensureOpen();
        return segment.byteSize();
    }

    @Override
    public synchronized void retain() {
        ensureOpen();
        if (references == Integer.MAX_VALUE) {
            throw new IllegalStateException("Storage view reference count overflow");
        }
        references++;
    }

    @Override
    public void release() {
        boolean releaseParent = false;
        synchronized (this) {
            ensureOpen();
            if (--references == 0) {
                releaseParent = true;
            }
        }
        if (releaseParent) {
            parent.release();
        }
    }

    private void ensureOpen() {
        if (references <= 0) {
            throw new IllegalStateException("Storage view has already been released");
        }
    }
}
