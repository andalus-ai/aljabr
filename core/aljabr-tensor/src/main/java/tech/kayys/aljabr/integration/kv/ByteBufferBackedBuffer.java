package tech.kayys.aljabr.integration.kv;

import tech.kayys.aljabr.core.memory.Buffer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Wraps a direct ByteBuffer as an aljabr Buffer. Attempts to create a
 * zero-copy MemorySegment
 * view of the ByteBuffer using MemorySegment.ofByteBuffer via reflection. Falls
 * back to allocating
 * a new Arena+segment and copying the contents if necessary.
 * @author bhangun
 */
public final class ByteBufferBackedBuffer implements Buffer {
    private final ByteBuffer byteBuffer;
    private final Arena arena;
    private final MemorySegment segment;
    private final boolean owned;
    private final AtomicInteger refCount = new AtomicInteger(1);
    private final Runnable onRelease;

    public ByteBufferBackedBuffer(ByteBuffer byteBuffer) {
        this(byteBuffer, () -> {
        });
    }

    public ByteBufferBackedBuffer(ByteBuffer byteBuffer, Runnable onRelease) {
        this.byteBuffer = Objects.requireNonNull(byteBuffer);
        this.onRelease = onRelease == null ? () -> {
        } : onRelease;
        Arena localArena = null;
        MemorySegment seg = null;
        boolean owns = false;
        try {
            // Try MemorySegment.ofByteBuffer(ByteBuffer)
            Method m = MemorySegment.class.getMethod("ofByteBuffer", ByteBuffer.class);
            seg = (MemorySegment) m.invoke(null, byteBuffer);
            localArena = Arena.ofShared();
            owns = false;
        } catch (Throwable t1) {
            try {
                // Try MemorySegment.ofByteBuffer(ByteBuffer, Arena)
                Method m2 = MemorySegment.class.getMethod("ofByteBuffer", ByteBuffer.class, Arena.class);
                localArena = Arena.ofShared();
                seg = (MemorySegment) m2.invoke(null, byteBuffer, localArena);
                owns = false;
            } catch (Throwable t2) {
                // Fallback: allocate new arena and copy
                localArena = Arena.ofShared();
                seg = localArena.allocate(byteBuffer.capacity());
                ByteBuffer dup = byteBuffer.duplicate();
                dup.rewind();
                seg.asByteBuffer().put(dup);
                seg.asByteBuffer().rewind();
                owns = true;
            }
        }
        this.arena = localArena;
        this.segment = seg;
        this.owned = owns;
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
        refCount.incrementAndGet();
    }

    @Override
    public synchronized void release() {
        if (refCount.decrementAndGet() == 0) {
            try {
                // notify external callback (e.g., to unpin cache page)
                onRelease.run();
            } catch (Throwable ignored) {
            }
            if (owned) {
                // close arena to free memory
                arena.close();
            }
        }
    }
}
