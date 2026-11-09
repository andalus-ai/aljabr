package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Proxy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.Tensor;

class MetalTensorStorageAccessTest {

    private static final class HostStorage implements HostAccessibleStorage {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment segment = arena.allocate(Float.BYTES);
        private int references = 1;

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
    }

    private static Tensor tensor(TensorStorage storage) {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> {
                    if ("storage".equals(method.getName())) {
                        return storage;
                    }
                    if ("toString".equals(method.getName())) {
                        return "StorageOnlyTensor";
                    }
                    throw new UnsupportedOperationException(
                            "Unexpected Tensor method in storage access test: " + method.getName());
                });
    }

    private static Tensor tensorWithoutStorage() {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> {
                    if ("storage".equals(method.getName())) {
                        throw new UnsupportedOperationException("Backing storage is not exposed");
                    }
                    if ("toString".equals(method.getName())) {
                        return "StorageUnavailableTensor";
                    }
                    throw new UnsupportedOperationException(
                            "Unexpected Tensor method in storage access test: " + method.getName());
                });
    }

    @Test
    void acceptsHostAccessibleStorageWithoutRequiringDefaultTensorOrBuffer() {
        HostStorage storage = new HostStorage();
        try {
            Tensor tensor = tensor(storage);

            assertSame(storage, MetalTensorStorageAccess.hostStorage(tensor));
            assertSame(storage.segment(), MetalTensorStorageAccess.segment(tensor));
        } finally {
            storage.release();
        }
    }

    @Test
    void rejectsDeviceLocalStorageUntilMetalHasADeviceNativeBindingContract() {
        Tensor tensor = tensor(new DeviceOnlyStorage());

        UnsupportedOperationException error = assertThrows(
                UnsupportedOperationException.class,
                () -> MetalTensorStorageAccess.segment(tensor));

        assertTrue(error.getMessage().contains("requires host-accessible tensor storage"));
        assertTrue(error.getMessage().contains("Metal-specific binding and allocator"));
    }

    @Test
    void rejectsTensorImplementationsThatDoNotExposeStorage() {
        Tensor tensor = tensorWithoutStorage();

        assertThrows(UnsupportedOperationException.class,
                () -> MetalTensorStorageAccess.segment(tensor));
    }
}
