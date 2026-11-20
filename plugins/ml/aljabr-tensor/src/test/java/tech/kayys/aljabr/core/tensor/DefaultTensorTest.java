package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Proxy;
import java.util.concurrent.CompletableFuture;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
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
    void backwardPopulatesUnitGradientForDifferentiableScalarLeaf() {
        CpuBackend backend = new CpuBackend();
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, backend);
        try {
            tensor.setRequiresGrad(true);
            tensor.backward();
            assertEquals(1.0f, tensor.grad().item(), 1.0e-6f);
        } finally {
            tensor.release();
            backend.close();
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
    void tensorViewRemainsUsableAfterParentTensorIsReleased() {
        CpuBuffer parentBuffer = new CpuBuffer(2L * Float.BYTES);
        parentBuffer.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 2.5f);
        parentBuffer.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, Float.BYTES, 9.0f);

        CpuBuffer viewBuffer = CpuBuffer.view(parentBuffer, 0L, Float.BYTES);
        var parentSegment = parentBuffer.segment();
        DefaultTensor parent = new DefaultTensor(
                new Shape(2), DType.F32, DeviceType.CPU, parentBuffer, unusedBackend());
        DefaultTensor view = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.CPU, viewBuffer, unusedBackend());

        parent.release();

        try {
            assertEquals(2.5f, view.item(), 0f);
        } finally {
            view.release();
        }

        assertFalse(parentSegment.scope().isAlive());
    }

    @Test
    void replacingAndClearingGradientReleasesPreviouslyOwnedGradient() {
        DefaultTensor owner = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL,
                new DeviceOnlyStorage(Float.BYTES), unusedBackend());
        DeviceOnlyStorage firstStorage = new DeviceOnlyStorage(Float.BYTES);
        DeviceOnlyStorage replacementStorage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor firstGradient = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, firstStorage, unusedBackend());
        DefaultTensor replacementGradient = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, replacementStorage, unusedBackend());

        try {
            owner.setGrad(firstGradient);
            assertSame(firstGradient, owner.grad());
            assertEquals(1, firstStorage.references());

            owner.setGrad(replacementGradient);
            assertSame(replacementGradient, owner.grad());
            assertEquals(0, firstStorage.references());
            assertEquals(1, replacementStorage.references());

            owner.setGrad(null);
            assertNull(owner.grad());
            assertEquals(0, replacementStorage.references());
        } finally {
            owner.release();
        }
    }

    @Test
    void rejectsOwningItselfAsGradientWithoutChangingOwnership() {
        DeviceOnlyStorage storage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, storage, unusedBackend());

        try {
            assertThrows(IllegalArgumentException.class, () -> tensor.setGrad(tensor));
            assertNull(tensor.grad());
            assertEquals(1, storage.references());
        } finally {
            tensor.release();
        }
    }

    @Test
    void replacementRemainsOwnedWhenPreviousGradientCleanupFails() {
        DeviceOnlyStorage ownerStorage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor owner = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, ownerStorage, unusedBackend());
        DeviceOnlyStorage replacementStorage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor replacement = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, replacementStorage, unusedBackend());
        IllegalStateException cleanupFailure = new IllegalStateException("previous gradient cleanup failed");
        Tensor failingPreviousGradient = (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("release")) {
                        throw cleanupFailure;
                    }
                    if (method.getName().equals("toString")) {
                        return "FailingPreviousGradient";
                    }
                    throw new UnsupportedOperationException("Unexpected gradient method: " + method.getName());
                });

        try {
            owner.setGrad(failingPreviousGradient);

            IllegalStateException thrown = assertThrows(
                    IllegalStateException.class, () -> owner.setGrad(replacement));

            assertSame(cleanupFailure, thrown);
            assertSame(replacement, owner.grad(),
                    "The replacement is installed before cleanup of the previous gradient");
            assertEquals(1, replacementStorage.references(),
                    "The owner must retain the replacement even when previous-gradient cleanup fails");
        } finally {
            owner.release();
        }

        assertEquals(0, ownerStorage.references());
        assertEquals(0, replacementStorage.references(),
                "Releasing the owner must release the replacement whose ownership transferred");
    }

    @Test
    void releasesOwnedStorageWhenGradientCleanupFails() {
        DeviceOnlyStorage storage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor owner = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, storage, unusedBackend());
        IllegalStateException cleanupFailure = new IllegalStateException("gradient cleanup failed");
        Tensor failingGradient = (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("release")) {
                        throw cleanupFailure;
                    }
                    if (method.getName().equals("toString")) {
                        return "FailingGradient";
                    }
                    throw new UnsupportedOperationException("Unexpected gradient method: " + method.getName());
                });

        owner.setGrad(failingGradient);

        IllegalStateException thrown = assertThrows(IllegalStateException.class, owner::release);

        assertSame(cleanupFailure, thrown);
        assertEquals(0, storage.references(),
                "The tensor's storage must be released even when gradient cleanup fails");
        assertNull(owner.grad(), "The owned gradient reference must be cleared during release");
        assertDoesNotThrow(owner::release, "A repeated release must remain idempotent");
        assertEquals(0, storage.references());
    }

    @Test
    void rejectsStorageSmallerThanTensorWithoutTakingOwnership() {
        DeviceOnlyStorage storage = new DeviceOnlyStorage(Float.BYTES - 1L);
        assertThrows(IllegalArgumentException.class,
                () -> new DefaultTensor(new Shape(1), DType.F32, DeviceType.METAL, storage, unusedBackend()));
        assertEquals(1, storage.references());
        storage.release();
    }

    @Test
    void rejectsGradientAssignmentAfterReleaseWithoutTakingGradientOwnership() {
        DefaultTensor owner = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL,
                new DeviceOnlyStorage(Float.BYTES), unusedBackend());
        DeviceOnlyStorage gradientStorage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor gradient = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, gradientStorage, unusedBackend());

        owner.release();
        try {
            assertThrows(IllegalStateException.class, () -> owner.setGrad(gradient));
            assertEquals(1, gradientStorage.references());
        } finally {
            gradient.release();
        }

        assertEquals(0, gradientStorage.references());
    }

    @Test
    void concurrentReleaseReleasesOwnedStorageExactlyOnce() {
        DeviceOnlyStorage storage = new DeviceOnlyStorage(Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.METAL, storage, unusedBackend());

        CompletableFuture<?>[] releases = IntStream.range(0, 32)
                .mapToObj(ignored -> CompletableFuture.runAsync(tensor::release))
                .toArray(CompletableFuture[]::new);

        assertDoesNotThrow(() -> CompletableFuture.allOf(releases).join());
        assertEquals(0, storage.references());
        assertThrows(IllegalStateException.class, tensor::storage);
    }

}
