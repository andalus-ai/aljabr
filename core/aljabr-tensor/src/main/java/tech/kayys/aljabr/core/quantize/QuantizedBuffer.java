package tech.kayys.aljabr.core.quantize;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import tech.kayys.aljabr.core.memory.Buffer;
import tech.kayys.aljabr.core.tensor.DType;

/**
 * A buffer specifically designed for storing block-quantized memory.
 * 
 * <p>
 * Unlike standard uncompressed buffers, quantized buffers must keep track of
 * their underlying quantization format (e.g., Q8_0, Q4_K) to properly decode
 * blocks during execution.
 * @author bhangun
 */
public final class QuantizedBuffer implements Buffer {

    private final MemorySegment segment;
    private final Arena arena;
    private final DType format;
    private int refCount = 1;

    public QuantizedBuffer(MemorySegment segment, Arena arena, DType format) {
        this.segment = segment;
        this.arena = arena;
        this.format = format;
    }

    public DType format() {
        return format;
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
        // Quantized buffers are often loaded from disk (mmap) or pooled,
        // so their lifecycle is typically managed externally.
        // We decrement ref count, but do not close the arena directly here
        // unless explicitly owned. For simplicity in this implementation,
        // we assume the arena is managed by the unified memory store or pool.
        refCount--;
    }
}
