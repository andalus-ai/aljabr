package tech.kayys.aljabr.integration.kv;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;

/** Private reference-counted host allocation used by materialized snapshots. */
final class OwnedHostStorage implements HostAccessibleStorage {
    private final Arena arena;
    private final MemorySegment segment;
    private int references = 1;

    OwnedHostStorage(float[] values) {
        arena = Arena.ofShared();
        try {
            segment = arena.allocate(Math.multiplyExact((long) values.length, Float.BYTES), Float.BYTES);
            for (int index = 0; index < values.length; index++) {
                segment.set(ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES, values[index]);
            }
        } catch (RuntimeException | Error failure) {
            arena.close();
            throw failure;
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
            throw new IllegalStateException("Snapshot storage reference count overflow");
        }
        references++;
    }

    @Override
    public void release() {
        boolean closeArena = false;
        synchronized (this) {
            ensureOpen();
            if (--references == 0) {
                closeArena = true;
            }
        }
        if (closeArena) {
            arena.close();
        }
    }

    private void ensureOpen() {
        if (references <= 0) {
            throw new IllegalStateException("Snapshot storage has been released");
        }
    }
}
