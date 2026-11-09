package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.memory.TensorStorage;

class TensorFactoryTest {
    private static ComputeBackend cpuBackend() {
        try {
            return (ComputeBackend) Class.forName("tech.kayys.aljabr.backend.cpu.CpuBackend")
                    .getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("CPU test backend is unavailable", e);
        }
    }

    @Test void rejectsShapesTooLargeForJavaArraysBeforeAllocation() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.zeros(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.ones(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.randn(Long.MAX_VALUE));
    }
    @Test void rejectsDataWhoseLengthDoesNotMatchShape() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.of(new float[] {1f}, 2));
    }
    @Test void rejectsQuantizedStorageWithIncompleteBlockShape() {
        MemorySegment segment = Arena.ofAuto().allocate(16);
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.ofQuantized(segment, DType.Q8_0, 31));
    }
    @Test void quantizedTensorOwnsACopyOfTheRequiredStorageBytes() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(36);
            for (long i = 0; i < source.byteSize(); i++)
                source.set(java.lang.foreign.ValueLayout.JAVA_BYTE, i, (byte) (i + 1));
            Tensor tensor = TensorFactory.ofQuantized(source, cpuBackend(), DType.Q8_0, 32);
            try {
                assertEquals(DType.Q8_0, tensor.dtype());
                assertEquals(DeviceType.CPU, tensor.device());
                assertEquals(34, tensor.storage().sizeBytes());
                assertInstanceOf(CpuBuffer.class, tensor.storage());
                MemorySegment copy = ((CpuBuffer) tensor.storage()).segment();
                for (long i = 0; i < 34; i++)
                    assertEquals(source.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i),
                            copy.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i));
            } finally { tensor.release(); }
        }
    }
    @Test void quantizedTensorUsesBackendAllocationAndCopyContract() {
        AtomicLong allocated = new AtomicLong(-1), copied = new AtomicLong(-1), released = new AtomicLong();
        AtomicBoolean deviceRequested = new AtomicBoolean();
        TensorStorage opaque = new TensorStorage() {
            public long sizeBytes() { return 34; }
            public void retain() {}
            public void release() { released.incrementAndGet(); }
        };
        ComputeBackend delegate = cpuBackend();
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(), new Class<?>[] {ComputeBackend.class},
                (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "allocateStorage" -> { allocated.set((long) args[0]); return opaque; }
                        case "copyFromHost" -> { copied.set((long) args[2]); return null; }
                        case "deviceType" -> { deviceRequested.set(true); return DeviceType.METAL; }
                        default -> {
                            try { return method.invoke(delegate, args); }
                            catch (java.lang.reflect.InvocationTargetException e) { throw e.getCause(); }
                        }
                    }
                });
        try (Arena arena = Arena.ofConfined()) {
            Tensor tensor = TensorFactory.ofQuantized(arena.allocate(40), backend, DType.Q8_0, 32);
            try {
                assertEquals(34, allocated.get());
                assertEquals(34, copied.get());
                assertTrue(deviceRequested.get());
                assertEquals(DeviceType.METAL, tensor.device());
                assertSame(backend, tensor.backend());
                assertSame(opaque, tensor.storage());
            } finally { tensor.release(); }
        }
        assertEquals(1, released.get());
    }
    @Test void rejectsNonQuantizedDtypes() {
        try (Arena arena = Arena.ofConfined()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TensorFactory.ofQuantized(arena.allocate(128), cpuBackend(), DType.F32, 32));
        }
    }
    @Test void automaticBackendSelectionUsesCpuWhenMetalIsUnavailableAndDoesNotRequireHat() {
        Tensor tensor = TensorFactory.zeros(1);
        try { assertEquals("tech.kayys.aljabr.backend.cpu.CpuBackend", tensor.backend().getClass().getName()); }
        finally { tensor.release(); }
    }
}
