package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;

class MetalStorageAccessTest {

    @Test
    void nativeBridgeAcceptsHostAccessibleStorageWithoutLegacyBuffer() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(Float.BYTES);
            DefaultTensor tensor = new DefaultTensor(
                    new Shape(1), DType.F32, DeviceType.CPU,
                    new SegmentStorage(segment), new CpuBackend());

            assertSame(segment, MetalComputeBackend.metalSegment(tensor));
        }
    }

    @Test
    void nativeBridgeRejectsOpaqueDeviceStorageWithMetalSpecificCapabilityError() {
        TensorStorage opaque = new TensorStorage() {
            @Override
            public long sizeBytes() {
                return Float.BYTES;
            }

            @Override
            public void retain() {
            }

            @Override
            public void release() {
            }
        };
        DefaultTensor tensor = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, opaque, new CpuBackend());

        UnsupportedOperationException failure = assertThrows(
                UnsupportedOperationException.class,
                () -> MetalComputeBackend.metalSegment(tensor));

        assertEquals(
                "Metal FFM dispatch requires host-accessible storage; opaque device-local storage "
                        + "requires a Metal-native binding path",
                failure.getMessage());
    }

    private static final class SegmentStorage implements HostAccessibleStorage {
        private final MemorySegment segment;

        private SegmentStorage(MemorySegment segment) {
            this.segment = segment;
        }

        @Override
        public long sizeBytes() {
            return segment.byteSize();
        }

        @Override
        public MemorySegment segment() {
            return segment;
        }

        @Override
        public void retain() {
        }

        @Override
        public void release() {
        }
    }
}
