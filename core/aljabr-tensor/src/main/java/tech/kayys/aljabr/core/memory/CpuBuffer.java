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
    private int refCount = 1;

    public CpuBuffer(long sizeBytes) {
        this.arena = Arena.ofShared();
        this.segment = arena.allocate(sizeBytes);
        this.owned = true;
    }

    public CpuBuffer(MemorySegment segment, Arena arena) {
        this.segment = segment;
        this.arena = arena;
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
        refCount++;
    }

    @Override
    public synchronized void release() {
        if (--refCount == 0 && owned)
            arena.close();
    }
}
