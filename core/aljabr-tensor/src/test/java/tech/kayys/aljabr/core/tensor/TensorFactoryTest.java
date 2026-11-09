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
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorAllocator;
import tech.kayys.aljabr.core.memory.TensorStorage;

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
        var segment = Arena.ofAuto().allocate(16);
        assertThrows(IllegalArgumentException.class,
                () -> TensorFactory.ofQuantized(segment, DType.Q8_0, 31));
    }

    @Test
    void explicitBackendOwnsTensorAllocationAndDeviceIdentity() {
        ComputeBackend delegate = cpuBackend();
        AtomicLong allocatedBytes = new AtomicLong(-1);
        java.util.concurrent.atomic.AtomicBoolean deviceIdentityRequested = new java.util.concurrent.atomic.AtomicBoolean();
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("allocateStorage")) {
                        allocatedBytes.set((long) args[0]);
                        return delegate.allocate((long) args[0]);
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

        ComputeBackend delegate = cpuBackend();
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
    void quantizedTensorOwnsACopyOfTheRequiredStorageBytes() {
        try (Arena sourceArena = Arena.ofConfined()) {
            MemorySegment source = sourceArena.allocate(36);
            for (long i = 0; i < source.byteSize(); i++) {
                source.set(java.lang.foreign.ValueLayout.JAVA_BYTE, i, (byte) (i + 1));
            }

            Tensor tensor = TensorFactory.ofQuantized(source, cpuBackend(), DType.Q8_0, 32);
            try {
                assertEquals(DType.Q8_0, tensor.dtype());
                assertEquals(DeviceType.CPU, tensor.device());
                assertEquals(34, tensor.storage().sizeBytes());
                assertInstanceOf(CpuBuffer.class, tensor.storage());
                MemorySegment copied = ((CpuBuffer) tensor.storage()).segment();
                for (long i = 0; i < 34; i++) {
                    assertEquals(source.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i),
                            copied.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i));
                }
            } finally {
                tensor.release();
            }
        }
    }

    @Test
    void quantizedTensorUsesBackendAllocationAndCopyContract() {
        AtomicLong allocatedBytes = new AtomicLong(-1);
        AtomicLong copiedBytes = new AtomicLong(-1);
        AtomicBoolean deviceIdentityRequested = new AtomicBoolean();
        AtomicLong released = new AtomicLong();
        TensorStorage opaqueStorage = new TensorStorage() {
            @Override public long sizeBytes() { return 34; }
            @Override public void retain() { }
            @Override public void release() { released.incrementAndGet(); }
        };
        ComputeBackend delegate = cpuBackend();
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "allocateStorage" -> {
                            allocatedBytes.set((long) args[0]);
                            return opaqueStorage;
                        }
                        case "copyFromHost" -> {
                            copiedBytes.set((long) args[2]);
                            return null;
                        }
                        case "deviceType" -> {
                            deviceIdentityRequested.set(true);
                            return DeviceType.METAL;
                        }
                        default -> {
                            try {
                                return method.invoke(delegate, args);
                            } catch (java.lang.reflect.InvocationTargetException e) {
                                throw e.getCause();
                            }
                        }
                    }
                });

        try (Arena arena = Arena.ofConfined()) {
            Tensor tensor = TensorFactory.ofQuantized(arena.allocate(40), backend, DType.Q8_0, 32);
            try {
                assertEquals(34, allocatedBytes.get());
                assertEquals(34, copiedBytes.get());
                assertTrue(deviceIdentityRequested.get());
                assertEquals(DeviceType.METAL, tensor.device());
                assertSame(backend, tensor.backend());
                assertSame(opaqueStorage, tensor.storage());
            } finally {
                tensor.release();
            }
        }
        assertEquals(1L, released.get());
    }

    @Test
    void rejectsNonQuantizedDtypes() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(128);
            assertThrows(IllegalArgumentException.class,
                    () -> TensorFactory.ofQuantized(source, cpuBackend(), DType.F32, 32));
        }
    }

    @Test
    void deprecatedConvenienceFactoryRequiresAnExplicitBackend() {
        IllegalStateException failure = assertThrows(
                IllegalStateException.class, () -> TensorFactory.zeros(1));
        assertTrue(failure.getMessage().contains("pass it to the TensorFactory overload"));
    }

    @Test
    void deprecatedConvenienceFactoryStillValidatesArgumentsBeforeBackendRequirement() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.zeros(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.ones(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.randn(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.of(new float[] {1f}, 2));
        assertThrows(IllegalStateException.class, () -> TensorFactory.of(new float[] {1f}, 1));
        assertThrows(IllegalStateException.class, () -> TensorFactory.full(2f, 1));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(34);
            assertThrows(IllegalStateException.class,
                    () -> TensorFactory.ofQuantized(source, DType.Q8_0, 32));
        }
    }

    @Test
    void tensorFactoryAcceptsAllocationPolicyIndependentOfComputeBackend() {
        ComputeBackend backend = cpuBackend();
        AtomicLong requestedBytes = new AtomicLong(-1);
        AtomicLong released = new AtomicLong();
        TensorAllocator allocator = size -> {
            requestedBytes.set(size);
            return new HostAccessibleStorage() {
                private final Arena arena = Arena.ofShared();
                private final MemorySegment segment = arena.allocate(size);
                private boolean live = true;

                @Override public long sizeBytes() { return segment.byteSize(); }
                @Override public MemorySegment segment() { return segment; }
                @Override public void retain() {
                    if (!live) throw new IllegalStateException("released");
                }
                @Override public void release() {
                    if (live) {
                        live = false;
                        arena.close();
                        released.incrementAndGet();
                    }
                }
            };
        };

        Tensor tensor = TensorFactory.of(
                new float[] {1f, 2f, 3f, 4f}, backend, allocator, 2, 2);
        try {
            assertEquals(16L, requestedBytes.get());
            assertSame(backend, tensor.backend());
            assertEquals(DeviceType.CPU, tensor.device());
            assertEquals(16L, tensor.storage().sizeBytes());
        } finally {
            tensor.release();
        }
        assertEquals(1L, released.get());
    }

    @Test
    void quantizedFactoryAcceptsIndependentAllocationPolicy() {
        ComputeBackend backend = cpuBackend();
        AtomicLong requestedBytes = new AtomicLong(-1);
        TensorAllocator allocator = size -> {
            requestedBytes.set(size);
            return new CpuBuffer(size);
        };

        try (Arena arena = Arena.ofConfined()) {
            Tensor tensor = TensorFactory.ofQuantized(
                    arena.allocate(40), backend, allocator, DType.Q8_0, 32);
            try {
                assertEquals(34L, requestedBytes.get());
                assertEquals(DType.Q8_0, tensor.dtype());
                assertEquals(34L, tensor.storage().sizeBytes());
            } finally {
                tensor.release();
            }
        }
    }

    private static ComputeBackend cpuBackend() {
        try {
            return (ComputeBackend) Class.forName("tech.kayys.aljabr.backend.cpu.CpuBackend")
                    .getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("CPU test backend is unavailable", e);
        }
    }
}
