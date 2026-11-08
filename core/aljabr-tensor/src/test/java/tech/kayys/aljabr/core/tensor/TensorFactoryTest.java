package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicLong;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.TensorStorage;
import java.lang.foreign.MemorySegment;

class TensorFactoryTest {
    @Test
    void rejectsShapesTooLargeForJavaArraysBeforeAllocation() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.zeros(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.ones(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.randn(Long.MAX_VALUE));
    }

    @Test
    void rejectsDataWhoseLengthDoesNotMatchShape() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.of(new float[] {1f}, 2));
    }

    @Test
    void rejectsQuantizedStorageWithIncompleteBlockShape() {
        var segment = java.lang.foreign.Arena.ofAuto().allocate(16);
        assertThrows(IllegalArgumentException.class,
                () -> TensorFactory.ofQuantized(segment, DType.Q8_0, 31));
    }

    @Test
    void explicitBackendOwnsTensorAllocationAndDeviceIdentity() {
        ComputeBackend delegate;
        try {
            delegate = (ComputeBackend) Class.forName("tech.kayys.aljabr.backend.cpu.CpuBackend")
                    .getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("CPU test backend is unavailable", e);
        }
        AtomicLong allocatedBytes = new AtomicLong(-1);
        java.util.concurrent.atomic.AtomicBoolean deviceIdentityRequested = new java.util.concurrent.atomic.AtomicBoolean();
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("allocate")) {
                        allocatedBytes.set((long) args[0]);
                    }
                    if (method.getName().equals("deviceType")) {
                        deviceIdentityRequested.set(true);
                        return DeviceType.METAL;
                    }
                    try {
                        return method.invoke(delegate, args);
                    } catch (java.lang.reflect.InvocationTargetException e) {
                        throw e.getCause();
                    }
                });

        Tensor tensor = TensorFactory.zeros(backend, 2, 3);
        try {
            assertEquals(24L, allocatedBytes.get());
            assertEquals(DeviceType.METAL, tensor.device());
            assertTrue(deviceIdentityRequested.get());
            assertSame(backend, tensor.backend());
            assertEquals(24L, ((DefaultTensor) tensor).buffer().sizeBytes());
        } finally {
            tensor.release();
        }
    }



    @Test
    void defaultTensorCanOwnStorageWithoutHostMemorySegment() {
        AtomicLong released = new AtomicLong();
        TensorStorage deviceStorage = new TensorStorage() {
            private boolean live = true;

            @Override
            public long sizeBytes() {
                return 24;
            }

            @Override
            public void retain() {
                if (!live) throw new IllegalStateException("released");
            }

            @Override
            public void release() {
                live = false;
                released.incrementAndGet();
            }
        };

        ComputeBackend delegate;
        try {
            delegate = (ComputeBackend) Class.forName("tech.kayys.aljabr.backend.cpu.CpuBackend")
                    .getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("CPU test backend is unavailable", e);
        }
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("allocateStorage")) return deviceStorage;
                    if (method.getName().equals("copyFromHost")) return null;
                    try {
                        return method.invoke(delegate, args);
                    } catch (java.lang.reflect.InvocationTargetException e) {
                        throw e.getCause();
                    }
                });

        Tensor tensor = TensorFactory.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, backend, 2, 3);
        assertSame(deviceStorage, ((DefaultTensor) tensor).storage());
        assertThrows(UnsupportedOperationException.class, () -> ((DefaultTensor) tensor).buffer());
        assertThrows(UnsupportedOperationException.class, tensor::toFloatArray);
        tensor.release();
        assertEquals(1L, released.get());
    }

    @Test
    void automaticBackendSelectionUsesCpuWhenMetalIsUnavailableAndDoesNotRequireHat() {
        Tensor tensor = TensorFactory.zeros(1);
        assertEquals("tech.kayys.aljabr.backend.cpu.CpuBackend", tensor.backend().getClass().getName());
    }

}
