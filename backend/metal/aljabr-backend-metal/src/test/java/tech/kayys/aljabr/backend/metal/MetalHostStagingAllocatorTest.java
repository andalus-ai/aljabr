package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.DeviceType;

class MetalHostStagingAllocatorTest {

    @Test
    void metalOutputsUseConfiguredHostStagingAllocator() {
        AtomicInteger allocations = new AtomicInteger();
        MetalComputeBackend backend = new MetalComputeBackend(size -> {
            allocations.incrementAndGet();
            return new TrackingHostStorage(size);
        });

        Tensor result = backend.zerosLike(new tech.kayys.aljabr.core.tensor.DefaultTensor(
                new Shape(4), DType.F32, DeviceType.CPU,
                new TrackingHostStorage(4L * Float.BYTES), backend));

        assertEquals(1, allocations.get());
        assertInstanceOf(HostAccessibleStorage.class, result.storage());
        assertEquals(4L * Float.BYTES, result.storage().sizeBytes());
        result.release();
    }

    @Test
    void rejectsAndReleasesOpaqueDeviceStorage() {
        AtomicInteger releases = new AtomicInteger();
        MetalComputeBackend backend = new MetalComputeBackend(size -> new TensorStorage() {
            @Override public long sizeBytes() { return size; }
            @Override public void retain() { }
            @Override public void release() { releases.incrementAndGet(); }
        });
        Tensor input = new tech.kayys.aljabr.core.tensor.DefaultTensor(
                new Shape(1), DType.F32, DeviceType.CPU,
                new TrackingHostStorage(Float.BYTES), backend);

        assertThrows(IllegalArgumentException.class, () -> backend.zerosLike(input));
        assertEquals(1, releases.get());
        input.release();
    }

    private static final class TrackingHostStorage implements HostAccessibleStorage {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment segment;
        private int references = 1;

        private TrackingHostStorage(long sizeBytes) {
            segment = arena.allocate(sizeBytes);
        }

        @Override public MemorySegment segment() { return segment; }
        @Override public long sizeBytes() { return segment.byteSize(); }
        @Override public synchronized void retain() {
            if (references <= 0) throw new IllegalStateException("released");
            references++;
        }
        @Override public synchronized void release() {
            if (references <= 0) throw new IllegalStateException("released");
            if (--references == 0) arena.close();
        }
    }
}
