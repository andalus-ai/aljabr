package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;

class CpuBackendStorageTest {

    @Test
    void cpuKernelsAcceptHostAccessibleStorageWithoutLegacyBuffer() {
        CpuBackend backend = new CpuBackend();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment leftMemory = arena.allocate(2L * Float.BYTES);
            MemorySegment rightMemory = arena.allocate(2L * Float.BYTES);
            leftMemory.set(ValueLayout.JAVA_FLOAT, 0, 1.5f);
            leftMemory.set(ValueLayout.JAVA_FLOAT, Float.BYTES, 2.5f);
            rightMemory.set(ValueLayout.JAVA_FLOAT, 0, 3.0f);
            rightMemory.set(ValueLayout.JAVA_FLOAT, Float.BYTES, 4.0f);

            DefaultTensor left = new DefaultTensor(
                    new Shape(2), DType.F32, DeviceType.CPU,
                    new SegmentStorage(leftMemory), backend);
            DefaultTensor right = new DefaultTensor(
                    new Shape(2), DType.F32, DeviceType.CPU,
                    new SegmentStorage(rightMemory), backend);

            var result = backend.add(left, right);

            assertArrayEquals(new float[] { 4.5f, 6.5f },
                    result.toFloatArray(), 0.0001f);
            result.release();
        }
    }

    @Test
    void cpuKernelsRejectDeviceLocalStorageWithCapabilityError() {
        CpuBackend backend = new CpuBackend();
        TensorStorage deviceStorage = new TensorStorage() {
            @Override
            public long sizeBytes() {
                return 2L * Float.BYTES;
            }

            @Override
            public void retain() {
            }

            @Override
            public void release() {
            }
        };
        DefaultTensor input = new DefaultTensor(
                new Shape(2), DType.F32, DeviceType.CPU, deviceStorage, backend);

        assertThrows(UnsupportedOperationException.class,
                () -> backend.mul(input, 2.0f));
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
