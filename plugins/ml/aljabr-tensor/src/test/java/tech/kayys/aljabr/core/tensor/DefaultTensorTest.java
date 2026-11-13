package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Proxy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;

class DefaultTensorTest {
    private static ComputeBackend unusedBackend() {
        return (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    throw new UnsupportedOperationException("Backend operations are not used by this test");
                });
    }

    private static final class HostOnlyStorage implements HostAccessibleStorage {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment segment;
        private int references = 1;

        private HostOnlyStorage(long sizeBytes) {
            this.segment = arena.allocate(sizeBytes);
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
            if (references <= 0) {
                throw new IllegalStateException("Storage has been released");
            }
            references++;
        }

        @Override
        public synchronized void release() {
            if (references <= 0) {
                throw new IllegalStateException("Storage has already been released");
            }
            if (--references == 0) {
                arena.close();
            }
        }
    }

    private static final class DeviceOnlyStorage implements TensorStorage {
        private final long sizeBytes;
        private int references = 1;

        private DeviceOnlyStorage(long sizeBytes) {
            this.sizeBytes = sizeBytes;
        }

        @Override
        public long sizeBytes() {
            return sizeBytes;
        }

        @Override
        public synchronized void retain() {
            if (references <= 0) {
                throw new IllegalStateException("Storage has been released");
            }
            references++;
        }

        @Override
        public synchronized void release() {
            if (references <= 0) {
                throw new IllegalStateException("Storage has already been released");
            }
            references--;
        }

        private synchronized int references() {
            return references;
        }
    }

    @Test
    void itemReturnsActualScalarValue() {
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        buffer.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 3.5f);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, unusedBackend());
        try {
            assertEquals(3.5f, tensor.item(), 0f);
        } finally {
            tensor.release();
        }
    }

    @Test
    void itemRejectsNonScalarTensors() {
        CpuBuffer buffer = new CpuBuffer(2L * Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(2), DType.F32, DeviceType.CPU, buffer, unusedBackend());
        try {
            assertThrows(IllegalStateException.class, tensor::item);
        } finally {
            tensor.release();
        }
    }

    @Test
    void floatArrayExportRejectsQuantizedDataInsteadOfReinterpretingBytes() {
        CpuBuffer buffer = new CpuBuffer(18);
        DefaultTensor tensor = new DefaultTensor(new Shape(32), DType.Q4_0, DeviceType.CPU, buffer, unusedBackend());
        try {
            assertThrows(UnsupportedOperationException.class, tensor::toFloatArray);
        } finally {
            tensor.release();
        }
    }

    @Test
    void exposesNeutralStorageAndPreservesLegacyBufferAccessor() {
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, unusedBackend());
        try {
            Tensor publicTensor = tensor;
            assertSame(buffer, publicTensor.storage());
            assertSame(buffer, tensor.buffer());
            assertEquals(Float.BYTES, publicTensor.storage().sizeBytes());
        } finally {
            tensor.release();
        }
    }

    @Test
    void readsHostAccessibleStorageWithoutRequiringLegacyBuffer() {
        HostOnlyStorage storage = new HostOnlyStorage(Float.BYTES);
        storage.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 7.25f);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU,
                (TensorStorage) storage, unusedBackend());
        try {
            assertSame(storage, tensor.storage());
            assertEquals(7.25f, tensor.item(), 0f);
            assertEquals(7.25f, tensor.toFloatArray()[0], 0f);
            assertThrows(UnsupportedOperationException.class, tensor::buffer);
        } finally {
            tensor.release();
        }
    }

    @Test
    void supportsDeviceLocalStorageWithoutAssumingHostMemory() {
        DeviceOnlyStorage storage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.METAL,
                storage, unusedBackend());
        try {
            assertSame(storage, tensor.storage());
            assertThrows(UnsupportedOperationException.class, tensor::toFloatArray);
            assertThrows(UnsupportedOperationException.class, tensor::item);
            assertThrows(UnsupportedOperationException.class, tensor::buffer);
        } finally {
            tensor.release();
        }
        assertEquals(0, storage.references());
    }


    @Test
    void backwardFailsExplicitlyWhenGradientsAreRequired() {
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, unusedBackend());
        try {
            tensor.setRequiresGrad(true);
            UnsupportedOperationException error = assertThrows(UnsupportedOperationException.class, tensor::backward);
            assertTrue(error.getMessage().contains("Automatic differentiation is not available"));
        } finally {
            tensor.release();
        }
    }

    @Test
    void backwardIsNoOpWhenGradientsAreNotRequired() {
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, unusedBackend());
        try {
            tensor.setRequiresGrad(false);
            assertDoesNotThrow(() -> {
                tensor.backward();
            });
        } finally {
            tensor.release();
        }
    }

    @Test
    void closeReleasesOwnedStorageExactlyOnceAndRejectsFurtherStorageAccess() {
        DeviceOnlyStorage storage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.METAL,
                storage, unusedBackend());

        tensor.close();
        tensor.close();
        tensor.release();

        assertEquals(0, storage.references());
        assertThrows(IllegalStateException.class, tensor::storage);
        assertThrows(IllegalStateException.class, tensor::toFloatArray);
    }

    @Test
    void rejectsStorageSmallerThanTensorWithoutTakingOwnership() {
        DeviceOnlyStorage storage = new DeviceOnlyStorage(Float.BYTES - 1L);
        assertThrows(IllegalArgumentException.class,
                () -> new DefaultTensor(new Shape(1), DType.F32, DeviceType.METAL, storage, unusedBackend()));
        assertEquals(1, storage.references());
        storage.release();
    }
}
