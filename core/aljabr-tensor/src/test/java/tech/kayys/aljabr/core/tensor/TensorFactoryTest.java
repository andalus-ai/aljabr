package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicLong;
import tech.kayys.aljabr.core.backend.ComputeBackend;

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
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("allocate")) {
                        allocatedBytes.set((long) args[0]);
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
            assertEquals(DeviceType.CPU, tensor.device());
            assertSame(backend, tensor.backend());
            assertEquals(24L, ((DefaultTensor) tensor).buffer().sizeBytes());
        } finally {
            tensor.release();
        }
    }


    @Test
    void automaticBackendSelectionUsesCpuWhenMetalIsUnavailableAndDoesNotRequireHat() {
        Tensor tensor = TensorFactory.zeros(1);
        assertEquals("tech.kayys.aljabr.backend.cpu.CpuBackend", tensor.backend().getClass().getName());
    }

}
