package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

class CpuBackendStorageTest {

    private static final class HostStorage implements HostAccessibleStorage {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment segment;
        private int references = 1;

        private HostStorage(float... values) {
            segment = arena.allocate((long) values.length * Float.BYTES);
            for (int i = 0; i < values.length; i++) {
                segment.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
            }
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

        private synchronized int references() {
            return references;
        }
    }

    private final CpuBackend backend = new CpuBackend();

    private DefaultTensor tensor(HostStorage storage, long... shape) {
        return new DefaultTensor(
                new Shape(shape), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend);
    }

    @Test
    void flattenPreservesLogicalValuesAndStorageOwnershipForRegularScalarAndEmptyTensors() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f, 5f, 6f);
        DefaultTensor input = tensor(storage, 2, 3);
        Tensor flattened = null;
        try {
            flattened = backend.flatten(input);
            assertArrayEquals(new long[] { 6 }, flattened.shape().dims());
            assertArrayEquals(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, flattened.toFloatArray(), 0f);
            assertEquals(2, storage.references());
        } finally {
            if (flattened != null) flattened.release();
            input.release();
        }
        assertEquals(0, storage.references());

        HostStorage scalarStorage = new HostStorage(7f);
        DefaultTensor scalar = tensor(scalarStorage);
        Tensor scalarFlat = null;
        try {
            scalarFlat = backend.flatten(scalar);
            assertArrayEquals(new long[] { 1 }, scalarFlat.shape().dims());
            assertArrayEquals(new float[] { 7f }, scalarFlat.toFloatArray(), 0f);
        } finally {
            if (scalarFlat != null) scalarFlat.release();
            scalar.release();
        }
        assertEquals(0, scalarStorage.references());

        HostStorage emptyStorage = new HostStorage();
        DefaultTensor empty = tensor(emptyStorage, 2, 0, 3);
        Tensor emptyFlat = null;
        try {
            emptyFlat = backend.flatten(empty);
            assertArrayEquals(new long[] { 0 }, emptyFlat.shape().dims());
            assertEquals(0, emptyFlat.numel());
        } finally {
            if (emptyFlat != null) emptyFlat.release();
            empty.release();
        }
        assertEquals(0, emptyStorage.references());
    }

    @Test
    void broadcastOperationsReadStridedOperandsWithStorageOffsets() {
        HostStorage leftStorage = new HostStorage(99f, 1f, 99f, 2f);
        HostStorage rightStorage = new HostStorage(99f, 10f, 20f);
        DefaultTensor left = new DefaultTensor(
                new Shape(2, 1), DType.F32, DeviceType.CPU, (TensorStorage) leftStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 1), new long[] {2, 1}, 1));
        DefaultTensor right = new DefaultTensor(
                new Shape(1, 2), DType.F32, DeviceType.CPU, (TensorStorage) rightStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 2), new long[] {2, 1}, 1));
        Tensor added = null;
        Tensor multiplied = null;
        Tensor subtracted = null;
        Tensor divided = null;
        try {
            added = backend.add(left, right);
            multiplied = backend.mul(left, right);
            subtracted = backend.sub(left, right);
            divided = backend.div(left, right);

            assertArrayEquals(new float[] {11f, 21f, 12f, 22f}, added.toFloatArray(), 0f);
            assertArrayEquals(new float[] {10f, 20f, 20f, 40f}, multiplied.toFloatArray(), 0f);
            assertArrayEquals(new float[] {-9f, -19f, -8f, -18f}, subtracted.toFloatArray(), 0f);
            assertArrayEquals(new float[] {0.1f, 0.05f, 0.2f, 0.1f}, divided.toFloatArray(), 1e-6f);
        } finally {
            if (divided != null) divided.release();
            if (subtracted != null) subtracted.release();
            if (multiplied != null) multiplied.release();
            if (added != null) added.release();
            left.release();
            right.release();
        }
        assertEquals(0, leftStorage.references());
        assertEquals(0, rightStorage.references());
    }

    @Test
    void embeddingReadsStridedWeightsAndOffsetIntegerIndices() {
        HostStorage weightStorage = new HostStorage(
                99f, 1f, 99f, 2f, 99f, 3f, 99f, 4f, 99f, 5f, 99f, 6f);
        HostStorage indexStorage = new HostStorage(99f, 0f, 99f, 0f);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 0, 99);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 4, 2);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 8, 99);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 12, 0);
        DefaultTensor weight = new DefaultTensor(
                new Shape(3, 2), DType.F32, DeviceType.CPU, (TensorStorage) weightStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(3, 2), new long[] {4, 2}, 1));
        DefaultTensor indices = new DefaultTensor(
                new Shape(2), DType.I32, DeviceType.CPU, (TensorStorage) indexStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2), new long[] {2}, 1));
        Tensor result = null;
        try {
            result = backend.embedding(weight, indices, -1);
            assertArrayEquals(new long[] {2, 2}, result.shape().dims());
            assertArrayEquals(new float[] {5f, 6f, 1f, 2f}, result.toFloatArray(), 0f);
        } finally {
            if (result != null) result.release();
            indices.release();
            weight.release();
        }
        assertEquals(0, weightStorage.references());
        assertEquals(0, indexStorage.references());
    }

    @Test
    void catReadsStridedAndOffsetInputsInLogicalOrderAndReleasesStorage() {
        HostStorage firstStorage = new HostStorage(99f, 1f, 2f, 99f, 3f, 4f);
        HostStorage secondStorage = new HostStorage(5f, 7f, 6f, 8f);
        DefaultTensor first = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) firstStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] {3, 1}, 1));
        DefaultTensor second = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) secondStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] {1, 2}, 0));
        Tensor concatenated = null;
        try {
            concatenated = backend.cat(java.util.List.of(first, second), 0);
            assertArrayEquals(new long[] {4, 2}, concatenated.shape().dims());
            assertArrayEquals(
                    new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f},
                    concatenated.toFloatArray(), 0f);
        } finally {
            if (concatenated != null) concatenated.release();
            second.release();
            first.release();
        }
        assertEquals(0, firstStorage.references());
        assertEquals(0, secondStorage.references());
    }

    @Test
    void catReadsStridedInputsWhenConcatenatingAlongInnerAxis() {
        HostStorage firstStorage = new HostStorage(99f, 1f, 99f, 2f);
        HostStorage secondStorage = new HostStorage(3f, 99f, 4f);
        DefaultTensor first = new DefaultTensor(
                new Shape(2, 1), DType.F32, DeviceType.CPU, (TensorStorage) firstStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 1), new long[] {2, 1}, 1));
        DefaultTensor second = new DefaultTensor(
                new Shape(2, 1), DType.F32, DeviceType.CPU, (TensorStorage) secondStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 1), new long[] {2, 1}, 0));
        Tensor concatenated = null;
        try {
            concatenated = backend.cat(java.util.List.of(first, second), 1);
            assertArrayEquals(new long[] {2, 2}, concatenated.shape().dims());
            assertArrayEquals(new float[] {1f, 3f, 2f, 4f}, concatenated.toFloatArray(), 0f);
        } finally {
            if (concatenated != null) concatenated.release();
            second.release();
            first.release();
        }
        assertEquals(0, firstStorage.references());
        assertEquals(0, secondStorage.references());
    }

    @Test
    void cpuKernelsReadHostAccessibleStorageWithoutLegacyBuffer() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        Tensor result = null;
        try {
            result = backend.add(input, input);
            assertArrayEquals(new float[] { 2f, 4f, 6f, 8f }, result.toFloatArray(), 0f);
        } finally {
            if (result != null) {
                result.release();
            }
            input.release();
        }
        assertEquals(0, storage.references());
    }


    @Test
    void pointwiseArithmeticReadsNonZeroOffsetAndStridedLayouts() {
        HostStorage storage = new HostStorage(99f, 1f, 2f, 99f, 3f, 4f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 3, 1 }, 1));
        Tensor result = null;
        try {
            result = backend.add(input, input);
            assertArrayEquals(new float[] { 2f, 4f, 6f, 8f }, result.toFloatArray(), 0f);
        } finally {
            if (result != null) result.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void unaryActivationReadsTransposedLogicalLayout() {
        HostStorage storage = new HostStorage(-1f, 2f, -3f, 4f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 1, 2 }, 0));
        Tensor result = null;
        try {
            result = backend.relu(input);
            assertArrayEquals(new float[] { 0f, 0f, 2f, 4f }, result.toFloatArray(), 0f);
        } finally {
            if (result != null) result.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void reshapeSharesNeutralStorageWithIndependentOwnership() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        Tensor reshaped = backend.reshape(input, 4);
        try {
            assertSame(storage, reshaped.storage());
            assertEquals(2, storage.references());

            input.release();
            assertArrayEquals(new float[] { 1f, 2f, 3f, 4f }, reshaped.toFloatArray(), 0f);
        } finally {
            reshaped.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void reshapePreservesNonZeroOffsetOfContiguousSlice() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        Tensor slice = null;
        Tensor reshaped = null;
        try {
            slice = backend.slice(input, new long[] { 1, 0 }, new long[] { 1, 2 });
            reshaped = backend.reshape(slice, 2);
            assertSame(storage, reshaped.storage());
            assertEquals(2, reshaped.layout().storageOffset());
            assertArrayEquals(new float[] { 3f, 4f }, reshaped.toFloatArray(), 0f);
        } finally {
            if (reshaped != null) reshaped.release();
            if (slice != null) slice.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void contiguousSliceSharesAndRetainsNonBufferParent() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        Tensor slice = backend.slice(input, new long[] { 1, 0 }, new long[] { 1, 2 });
        try {
            assertSame(storage, slice.storage());
            assertInstanceOf(HostAccessibleStorage.class, slice.storage());
            assertEquals(2, storage.references());

            input.release();
            assertArrayEquals(new float[] { 3f, 4f }, slice.toFloatArray(), 0f);
        } finally {
            slice.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void nonContiguousSlicePreservesParentStorageAndLogicalOrder() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f, 5f, 6f);
        DefaultTensor input = tensor(storage, 2, 3);
        Tensor slice = null;
        try {
            slice = backend.slice(input, new long[] { 0, 1 }, new long[] { 2, 2 });
            assertSame(storage, slice.storage());
            assertEquals(2, storage.references());
            assertArrayEquals(new float[] { 2f, 3f, 5f, 6f }, slice.toFloatArray(), 0f);
        } finally {
            if (slice != null) {
                slice.release();
            }
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void sliceOfStridedInputPreservesInputStridesAndStorageOffset() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f, 5f, 6f);
        DefaultTensor input = new DefaultTensor(
                new Shape(3, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage,
                backend, new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(3, 2), new long[] { 1, 3 }, 0));
        Tensor slice = null;
        try {
            slice = backend.slice(input, new long[] { 1, 0 }, new long[] { 2, 2 });
            assertSame(storage, slice.storage());
            assertArrayEquals(new float[] { 2f, 5f, 3f, 6f }, slice.toFloatArray(), 0f);
            assertArrayEquals(new long[] { 1, 3 }, slice.layout().strides());
            assertEquals(1, slice.layout().storageOffset());
        } finally {
            if (slice != null) slice.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void bufferBackedContiguousSliceKeepsLegacyBufferAccessorWorking() {
        CpuBuffer buffer = new CpuBuffer(4L * Float.BYTES);
        for (int i = 0; i < 4; i++) {
            buffer.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, i + 1f);
        }
        DefaultTensor input = new DefaultTensor(new Shape(2, 2), DType.F32, DeviceType.CPU, buffer, backend);
        Tensor slice = null;
        try {
            slice = backend.slice(input, new long[] { 1, 0 }, new long[] { 1, 2 });
            assertInstanceOf(CpuBuffer.class, ((DefaultTensor) slice).buffer());
            assertArrayEquals(new float[] { 3f, 4f }, slice.toFloatArray(), 0f);
        } finally {
            if (slice != null) {
                slice.release();
            }
            input.release();
        }
    }

    @Test
    void flashAttentionReadsHostAccessibleStorageWithoutLegacyBuffer() {
        HostStorage q = new HostStorage(1f);
        HostStorage k = new HostStorage(1f);
        HostStorage v = new HostStorage(3f);
        DefaultTensor query = tensor(q, 1, 1);
        DefaultTensor key = tensor(k, 1, 1);
        DefaultTensor value = tensor(v, 1, 1);
        Tensor result = null;
        try {
            result = FlashAttentionCpu.forward(query, key, value, 1);
            assertArrayEquals(new float[] { 3f }, result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) {
                result.release();
            }
            query.release();
            key.release();
            value.release();
        }
        assertEquals(0, q.references());
        assertEquals(0, k.references());
        assertEquals(0, v.references());
    }

    @Test
    void reshapeRejectsElementCountChanges() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.reshape(input, 3));
            assertEquals(1, storage.references());
        } finally {
            input.release();
        }
    }

    @Test
    void sliceRejectsInvalidBoundsAndNonF32DtypesBeforeMemoryAccess() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.slice(input, new long[] { 1, 0 }, new long[] { 2, 1 }));
            assertThrows(IllegalArgumentException.class,
                    () -> backend.slice(input, new long[] { 0 }, new long[] { 1 }));
        } finally {
            input.release();
        }
        assertEquals(0, storage.references());

        HostStorage integerStorage = new HostStorage(1f);
        DefaultTensor integerTensor = new DefaultTensor(
                new Shape(1), DType.I32, DeviceType.CPU, (TensorStorage) integerStorage, backend);
        try {
            assertThrows(UnsupportedOperationException.class,
                    () -> backend.slice(integerTensor, new long[] { 0 }, new long[] { 1 }));
        } finally {
            integerTensor.release();
        }
        assertEquals(0, integerStorage.references());
    }
    @Test
    void ownedPoolClosesWithBackendButOutstandingTensorLeaseRemainsValid() {
        tech.kayys.aljabr.core.memory.OffHeapBufferPool pool =
                new tech.kayys.aljabr.core.memory.OffHeapBufferPool();
        CpuBackend owned = new CpuBackend(pool, CpuBackend.PoolOwnership.OWNED);
        CpuBuffer lease = owned.allocate(16);

        owned.close();
        owned.close();

        assertTrue(pool.isClosed());
        assertTrue(lease.segment().scope().isAlive());
        assertThrows(IllegalStateException.class, () -> owned.allocate(16));

        lease.release();
        assertFalse(pool.arena().scope().isAlive());
    }

    @Test
    void borrowedPoolIsNotClosedByBackend() {
        tech.kayys.aljabr.core.memory.OffHeapBufferPool pool =
                new tech.kayys.aljabr.core.memory.OffHeapBufferPool();
        CpuBackend borrowed = new CpuBackend(pool);

        CpuBuffer lease = borrowed.allocate(16);
        borrowed.close();

        assertFalse(pool.isClosed());
        assertTrue(lease.segment().scope().isAlive());
        assertThrows(IllegalStateException.class, () -> borrowed.allocate(16));

        lease.release();
        MemorySegment next = pool.acquire(16);
        pool.release(next);
        pool.close();
    }

    @Test
    void cpuKernelOutputsUseConfiguredTensorAllocator() {
        AtomicInteger allocations = new AtomicInteger();
        CpuBackend allocatingBackend = new CpuBackend(
                null,
                CpuBackend.PoolOwnership.BORROWED,
                size -> {
                    allocations.incrementAndGet();
                    return new CpuBuffer(size);
                });
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, allocatingBackend);
        Tensor result = null;
        try {
            result = allocatingBackend.add(input, input);
            assertEquals(1, allocations.get());
            assertInstanceOf(HostAccessibleStorage.class, result.storage());
            assertArrayEquals(new float[] {2f, 4f, 6f, 8f}, result.toFloatArray(), 0f);
        } finally {
            if (result != null) result.release();
            input.release();
            allocatingBackend.close();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void cpuBackendRejectsDeviceOnlyOutputAllocatorAndReleasesRejectedStorage() {
        AtomicInteger releases = new AtomicInteger();
        CpuBackend allocatingBackend = new CpuBackend(
                null,
                CpuBackend.PoolOwnership.BORROWED,
                size -> new TensorStorage() {
                    @Override public long sizeBytes() { return size; }
                    @Override public void retain() { }
                    @Override public void release() { releases.incrementAndGet(); }
                });

        assertThrows(IllegalArgumentException.class, () -> allocatingBackend.allocateStorage(16));
        assertEquals(1, releases.get());
        allocatingBackend.close();
    }

    @Test
    void reshapedContiguousSlicePreservesAliasingAndTransitiveOwnership() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        Tensor slice = null;
        Tensor reshaped = null;
        try {
            slice = backend.slice(input, new long[] { 1, 0 }, new long[] { 1, 2 });
            reshaped = backend.reshape(slice, 2);

            assertSame(slice.storage(), reshaped.storage());
            // Input, slice, and reshaped view each own one storage reference.
            assertEquals(3, storage.references());

            HostAccessibleStorage viewStorage = (HostAccessibleStorage) slice.storage();
            // The slice starts at physical element 2 in the shared base storage.
            viewStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 2L * Float.BYTES, 30f);
            assertArrayEquals(new float[] { 1f, 2f, 30f, 4f }, input.toFloatArray(), 0f);

            input.release();
            slice.release();

            assertEquals(1, storage.references());
            assertArrayEquals(new float[] { 30f, 4f }, reshaped.toFloatArray(), 0f);
        } finally {
            if (reshaped != null) {
                reshaped.release();
            }
            if (slice != null) {
                slice.release();
            }
            input.release();
        }
        assertEquals(0, storage.references());
    }


    @Test
    void globalReductionsReadStridedAndOffsetTensorLayouts() {
        HostStorage storage = new HostStorage(99f, 1f, 2f, 3f, 4f, 88f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 1, 2 }, 1));
        Tensor sum = null;
        Tensor mean = null;
        Tensor max = null;
        try {
            sum = backend.sum(input);
            mean = backend.mean(input);
            max = backend.max(input);

            assertArrayEquals(new float[] { 10f }, sum.toFloatArray(), 0f);
            assertArrayEquals(new float[] { 2.5f }, mean.toFloatArray(), 0f);
            assertArrayEquals(new float[] { 4f }, max.toFloatArray(), 0f);
        } finally {
            if (sum != null) sum.release();
            if (mean != null) mean.release();
            if (max != null) max.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void axisSumReadsTransposedLogicalOrderAndPreservesKeepDim() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 1, 2 }, 0));
        Tensor sumAxisZero = null;
        Tensor sumAxisOneKeepDim = null;
        try {
            sumAxisZero = backend.sum(input, 0, false);
            sumAxisOneKeepDim = backend.sum(input, -1, true);

            assertArrayEquals(new float[] { 3f, 7f }, sumAxisZero.toFloatArray(), 0f);
            assertArrayEquals(new long[] { 2 }, sumAxisZero.shape().dims());
            assertArrayEquals(new float[] { 4f, 6f }, sumAxisOneKeepDim.toFloatArray(), 0f);
            assertArrayEquals(new long[] { 2, 1 }, sumAxisOneKeepDim.shape().dims());
        } finally {
            if (sumAxisZero != null) sumAxisZero.release();
            if (sumAxisOneKeepDim != null) sumAxisOneKeepDim.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void axisMeanReadsNonZeroOffsetStridedTensor() {
        HostStorage storage = new HostStorage(77f, 1f, 2f, 3f, 4f, 88f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 1, 2 }, 1));
        Tensor mean = null;
        try {
            mean = backend.mean(input, -1, false);
            assertArrayEquals(new float[] { 2f, 3f }, mean.toFloatArray(), 0f);
        } finally {
            if (mean != null) mean.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }


    @Test
    void transposeNormalizesNegativeAxesAndPreservesLogicalValues() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f, 5f, 6f);
        DefaultTensor input = tensor(storage, 2, 3);
        Tensor trailing = null;
        Tensor sameAxis = null;
        try {
            trailing = input.transpose(-2, -1);
            sameAxis = input.transpose(-1, 1);
            assertArrayEquals(new long[] { 3, 2 }, trailing.shape().dims());
            assertArrayEquals(new float[] { 1f, 4f, 2f, 5f, 3f, 6f }, trailing.toFloatArray(), 0f);
            assertArrayEquals(new long[] { 2, 3 }, sameAxis.shape().dims());
            assertArrayEquals(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, sameAxis.toFloatArray(), 0f);
        } finally {
            if (sameAxis != null) sameAxis.release();
            if (trailing != null) trailing.release();
            input.release();
        }
        assertEquals(0, storage.references());

        HostStorage nonTerminalStorage = new HostStorage(
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f);
        DefaultTensor nonTerminalInput = tensor(nonTerminalStorage, 2, 2, 3);
        Tensor nonTerminal = null;
        Tensor defaultTranspose = null;
        try {
            nonTerminal = nonTerminalInput.transpose(0, -1);
            defaultTranspose = nonTerminalInput.transpose();
            assertArrayEquals(new long[] { 3, 2, 2 }, nonTerminal.shape().dims());
            assertArrayEquals(new float[] {
                    1f, 7f, 4f, 10f, 2f, 8f, 5f, 11f, 3f, 9f, 6f, 12f
            }, nonTerminal.toFloatArray(), 0f);
            assertArrayEquals(new long[] { 2, 3, 2 }, defaultTranspose.shape().dims());
        } finally {
            if (defaultTranspose != null) defaultTranspose.release();
            if (nonTerminal != null) nonTerminal.release();
            nonTerminalInput.release();
        }
        assertEquals(0, nonTerminalStorage.references());
    }

    @Test
    void transposeRejectsInvalidAxesBeforeAllocatingOutput() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> input.transpose(-3, 1));
            assertThrows(IllegalArgumentException.class, () -> input.transpose(0, 2));
            assertEquals(1, storage.references());
        } finally {
            input.release();
        }
        assertEquals(0, storage.references());
    }
    @Test
    void squeezeAndUnsqueezePreserveValuesAndUseSharedShapeContracts() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        Tensor trailing = null;
        Tensor leading = null;
        try {
            trailing = input.unsqueeze(-1);
            leading = input.unsqueeze(-3);
            assertArrayEquals(new long[] { 2, 2, 1 }, trailing.shape().dims());
            assertArrayEquals(new long[] { 1, 2, 2 }, leading.shape().dims());
            assertArrayEquals(new float[] { 1f, 2f, 3f, 4f }, trailing.toFloatArray(), 0f);
        } finally {
            if (leading != null) leading.release();
            if (trailing != null) trailing.release();
            input.release();
        }
        assertEquals(0, storage.references());

        HostStorage singletonStorage = new HostStorage(5f, 6f, 7f, 8f);
        DefaultTensor singletonInput = tensor(singletonStorage, 1, 2, 1, 2);
        Tensor squeezed = null;
        try {
            squeezed = singletonInput.squeeze();
            assertArrayEquals(new long[] { 2, 2 }, squeezed.shape().dims());
            assertArrayEquals(new float[] { 5f, 6f, 7f, 8f }, squeezed.toFloatArray(), 0f);
        } finally {
            if (squeezed != null) squeezed.release();
            singletonInput.release();
        }
        assertEquals(0, singletonStorage.references());
    }

    @Test
    void squeezeRetainsOneElementVectorForAllSingletonAndScalarInputs() {
        HostStorage singletonStorage = new HostStorage(9f);
        DefaultTensor singletonInput = tensor(singletonStorage, 1, 1);
        Tensor singletonResult = null;
        try {
            singletonResult = singletonInput.squeeze();
            assertArrayEquals(new long[] { 1 }, singletonResult.shape().dims());
        } finally {
            if (singletonResult != null) singletonResult.release();
            singletonInput.release();
        }
        assertEquals(0, singletonStorage.references());

        HostStorage scalarStorage = new HostStorage(11f);
        DefaultTensor scalarInput = tensor(scalarStorage);
        Tensor scalarResult = null;
        try {
            scalarResult = scalarInput.squeeze();
            assertArrayEquals(new long[] { 1 }, scalarResult.shape().dims());
            assertArrayEquals(new float[] { 11f }, scalarResult.toFloatArray(), 0f);
        } finally {
            if (scalarResult != null) scalarResult.release();
            scalarInput.release();
        }
        assertEquals(0, scalarStorage.references());
    }

    @Test
    void softmaxReadsStridedTensorWithNonZeroStorageOffset() {
        HostStorage storage = new HostStorage(99f, 1f, 2f, 3f, 4f, 88f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 1, 2 }, 1));
        Tensor result = null;
        try {
            result = backend.softmax(input, -1);
            float probabilityLow = (float) (1d / (1d + Math.exp(2d)));
            float probabilityHigh = 1f - probabilityLow;
            assertArrayEquals(
                    new float[] { probabilityLow, probabilityHigh, probabilityLow, probabilityHigh },
                    result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void logSoftmaxUsesStableLogSumExpForLargeLogits() {
        HostStorage storage = new HostStorage(1000f, 1001f);
        DefaultTensor input = tensor(storage, 2);
        Tensor result = null;
        try {
            result = backend.logSoftmax(input, 0);
            assertArrayEquals(
                    new float[] { -1.3132617f, -0.3132617f },
                    result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void batchNormInferenceReadsStridedInputWithStorageOffset() {
        HostStorage inputStorage = new HostStorage(99f, 1f, 2f, 3f, 4f, 88f);
        HostStorage weightStorage = new HostStorage(1f, 1f);
        HostStorage biasStorage = new HostStorage(0f, 0f);
        HostStorage meanStorage = new HostStorage(2f, 3f);
        HostStorage varianceStorage = new HostStorage(1f, 1f);
        DefaultTensor input = new DefaultTensor(
                new Shape(1, 2, 2), DType.F32, DeviceType.CPU, (TensorStorage) inputStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 2, 2), new long[] { 4, 1, 2 }, 1));
        DefaultTensor weight = tensor(weightStorage, 2);
        DefaultTensor bias = tensor(biasStorage, 2);
        DefaultTensor runningMean = tensor(meanStorage, 2);
        DefaultTensor runningVar = tensor(varianceStorage, 2);
        Tensor result = null;
        try {
            result = backend.batchNorm(input, weight, bias, runningMean, runningVar,
                    false, 0.1f, 0f);
            assertArrayEquals(new float[] { -1f, 1f, -1f, 1f }, result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            input.release();
            weight.release();
            bias.release();
            runningMean.release();
            runningVar.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, weightStorage.references());
        assertEquals(0, biasStorage.references());
        assertEquals(0, meanStorage.references());
        assertEquals(0, varianceStorage.references());
    }

    @Test
    void batchNormTrainingUsesBatchStatisticsAndUpdatesRunningStatistics() {
        HostStorage inputStorage = new HostStorage(1f, 10f, 3f, 14f);
        HostStorage meanStorage = new HostStorage(0f, 0f);
        HostStorage varianceStorage = new HostStorage(1f, 1f);
        DefaultTensor input = tensor(inputStorage, 2, 2);
        DefaultTensor runningMean = tensor(meanStorage, 2);
        DefaultTensor runningVar = tensor(varianceStorage, 2);
        Tensor result = null;
        try {
            result = backend.batchNorm(input, null, null, runningMean, runningVar,
                    true, 0.1f, 0f);
            assertArrayEquals(new float[] { -1f, -1f, 1f, 1f }, result.toFloatArray(), 1e-6f);
            assertArrayEquals(new float[] { 0.2f, 1.2f }, runningMean.toFloatArray(), 1e-6f);
            assertArrayEquals(new float[] { 1.1f, 1.7f }, runningVar.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            input.release();
            runningMean.release();
            runningVar.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, meanStorage.references());
        assertEquals(0, varianceStorage.references());
    }

    @Test
    void batchNormWithoutRunningStatisticsUsesBatchStatisticsInEvaluation() {
        HostStorage inputStorage = new HostStorage(1f, 10f, 3f, 14f);
        DefaultTensor input = tensor(inputStorage, 2, 2);
        Tensor result = null;
        try {
            result = backend.batchNorm(input, null, null, null, null,
                    false, 0.1f, 0f);
            assertArrayEquals(new float[] { -1f, -1f, 1f, 1f }, result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            input.release();
        }
        assertEquals(0, inputStorage.references());
    }

    @Test
    void batchNormRejectsInvalidMomentumAndChannelParameters() {
        HostStorage inputStorage = new HostStorage(1f, 2f, 3f, 4f);
        HostStorage parameterStorage = new HostStorage(1f);
        DefaultTensor input = tensor(inputStorage, 2, 2);
        DefaultTensor parameter = tensor(parameterStorage, 1);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.batchNorm(input, parameter, null, null, null,
                            false, 0.1f, 1e-5f));
            assertThrows(IllegalArgumentException.class,
                    () -> backend.batchNorm(input, null, null, null, null,
                            true, 1.1f, 1e-5f));
        } finally {
            input.release();
            parameter.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, parameterStorage.references());
    }

    @Test
    void layerNormReadsStridedInputWithStorageOffset() {
        HostStorage inputStorage = new HostStorage(99f, 1f, 2f, 3f, 4f, 88f);
        HostStorage weightStorage = new HostStorage(1f, 1f);
        HostStorage biasStorage = new HostStorage(0f, 0f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) inputStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 1, 2 }, 1));
        DefaultTensor weight = tensor(weightStorage, 2);
        DefaultTensor bias = tensor(biasStorage, 2);
        Tensor result = null;
        try {
            result = backend.layerNorm(input, new long[] { 2 }, weight, bias, 1e-5f);
            assertArrayEquals(new float[] { -1f, 1f, -1f, 1f }, result.toFloatArray(), 1e-5f);
        } finally {
            if (result != null) result.release();
            input.release();
            weight.release();
            bias.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, weightStorage.references());
        assertEquals(0, biasStorage.references());
    }

    @Test
    void rmsNormReadsStridedInputWithStorageOffset() {
        HostStorage inputStorage = new HostStorage(99f, 1f, 2f, 3f, 4f, 88f);
        HostStorage weightStorage = new HostStorage(1f, 1f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) inputStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] { 1, 2 }, 1));
        DefaultTensor weight = tensor(weightStorage, 2);
        Tensor result = null;
        try {
            result = backend.rmsNorm(input, weight, 0f);
            assertArrayEquals(new float[] {
                    (float) (1d / Math.sqrt(5d)), (float) (3d / Math.sqrt(5d)),
                    (float) (2d / Math.sqrt(10d)), (float) (4d / Math.sqrt(10d))
            }, result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            input.release();
            weight.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, weightStorage.references());
    }

    @Test
    void normalizationRejectsInvalidShapeAndEpsilon() {
        HostStorage inputStorage = new HostStorage(1f, 2f, 3f, 4f);
        HostStorage weightStorage = new HostStorage(1f, 1f);
        DefaultTensor input = tensor(inputStorage, 2, 2);
        DefaultTensor weight = tensor(weightStorage, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.layerNorm(input, new long[] { 3 }, weight, null, 1e-5f));
            assertThrows(IllegalArgumentException.class,
                    () -> backend.layerNorm(input, new long[] { 2 }, weight, null, Float.NaN));
            assertThrows(IllegalArgumentException.class,
                    () -> backend.rmsNorm(input, weight, -1f));
        } finally {
            input.release();
            weight.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, weightStorage.references());
    }


    @Test
    void spatialPoolingAndUpsamplingReadNonZeroOffsetTransposedLayouts() {
        HostStorage storage = new HostStorage(99f, 1f, 2f, 3f, 4f);
        DefaultTensor input = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] { 4, 4, 1, 2 }, 1));
        Tensor maximum = null;
        Tensor average = null;
        Tensor upsampled = null;
        try {
            maximum = backend.maxPool2d(input, 2, 1, 0);
            average = backend.adaptiveAvgPool2d(input, 1, 1);
            upsampled = backend.upsampleNearest2d(input, 4, 4);

            assertArrayEquals(new float[] { 4f }, maximum.toFloatArray(), 0f);
            assertArrayEquals(new float[] { 2.5f }, average.toFloatArray(), 0f);
            assertArrayEquals(new float[] {
                    1f, 1f, 3f, 3f,
                    1f, 1f, 3f, 3f,
                    2f, 2f, 4f, 4f,
                    2f, 2f, 4f, 4f
            }, upsampled.toFloatArray(), 0f);
        } finally {
            if (upsampled != null) upsampled.release();
            if (average != null) average.release();
            if (maximum != null) maximum.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void spatialPoolingRejectsInvalidKernelAndOutputDimensions() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 1, 1, 2, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.maxPool2d(input, 0, 1, 0));
            assertThrows(IllegalArgumentException.class, () -> backend.adaptiveAvgPool2d(input, 0, 1));
            assertThrows(IllegalArgumentException.class, () -> backend.upsampleNearest2d(input, 1, 0));
        } finally {
            input.release();
        }
        assertEquals(0, storage.references());
    }


    @Test
    void conv2dReadsStridedInputWeightAndBiasWithStorageOffsets() {
        HostStorage inputStorage = new HostStorage(99f, 1f, 2f, 3f, 4f);
        HostStorage weightStorage = new HostStorage(99f, 1f, 0f, 0f, 1f);
        HostStorage biasStorage = new HostStorage(99f, 2f);
        DefaultTensor input = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU, (TensorStorage) inputStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] { 4, 4, 1, 2 }, 1));
        DefaultTensor weight = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU, (TensorStorage) weightStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] { 4, 4, 1, 2 }, 1));
        DefaultTensor bias = new DefaultTensor(
                new Shape(1), DType.F32, DeviceType.CPU, (TensorStorage) biasStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1), new long[] { 1 }, 1));
        Tensor result = null;
        try {
            result = backend.conv2d(input, weight, bias, 1, 0, 1, 1);
            assertArrayEquals(new long[] { 1, 1, 1, 1 }, result.shape().dims());
            assertArrayEquals(new float[] { 7f }, result.toFloatArray(), 0f);
        } finally {
            if (result != null) result.release();
            input.release();
            weight.release();
            bias.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, weightStorage.references());
        assertEquals(0, biasStorage.references());
    }

    @Test
    void conv2dPartitionsInputChannelsForGroupedConvolution() {
        HostStorage inputStorage = new HostStorage(2f, 3f);
        HostStorage weightStorage = new HostStorage(4f, 5f);
        DefaultTensor input = tensor(inputStorage, 1, 2, 1, 1);
        DefaultTensor weight = tensor(weightStorage, 2, 1, 1, 1);
        Tensor result = null;
        try {
            result = backend.conv2d(input, weight, null, 1, 0, 1, 2);
            assertArrayEquals(new long[] { 1, 2, 1, 1 }, result.shape().dims());
            assertArrayEquals(new float[] { 8f, 15f }, result.toFloatArray(), 0f);
        } finally {
            if (result != null) result.release();
            input.release();
            weight.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, weightStorage.references());
    }

    @Test
    void conv2dRejectsInvalidGroupsBiasAndKernelGeometry() {
        HostStorage inputStorage = new HostStorage(1f, 2f, 3f, 4f);
        HostStorage weightStorage = new HostStorage(1f, 1f, 1f, 1f);
        HostStorage biasStorage = new HostStorage(1f, 2f);
        DefaultTensor input = tensor(inputStorage, 1, 1, 2, 2);
        DefaultTensor weight = tensor(weightStorage, 1, 1, 2, 2);
        DefaultTensor bias = tensor(biasStorage, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.conv2d(input, weight, null, 1, 0, 1, 2));
            assertThrows(IllegalArgumentException.class,
                    () -> backend.conv2d(input, weight, bias, 1, 0, 1, 1));
            assertThrows(IllegalArgumentException.class,
                    () -> backend.conv2d(input, weight, null, 0, 0, 1, 1));
            assertThrows(IllegalArgumentException.class,
                    () -> backend.conv2d(input, weight, null, 1, 0, 2, 1));
        } finally {
            input.release();
            weight.release();
            bias.release();
        }
        assertEquals(0, inputStorage.references());
        assertEquals(0, weightStorage.references());
        assertEquals(0, biasStorage.references());
    }

    @Test
    void matmulReadsTransposedAndNonZeroOffsetMatrixViews() {
        HostStorage leftStorage = new HostStorage(99f, 1f, 2f, 3f, 4f);
        DefaultTensor left = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) leftStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] {2, 1}, 1));
        HostStorage rightStorage = new HostStorage(5f, 6f, 7f, 8f);
        DefaultTensor right = tensor(rightStorage, 2, 2);
        Tensor transposed = null;
        Tensor result = null;
        try {
            transposed = backend.transpose(left, 0, 1);
            result = backend.matmul(transposed, right);
            assertArrayEquals(new float[] {26f, 30f, 38f, 44f}, result.toFloatArray(), 0f);
        } finally {
            if (result != null) result.release();
            if (transposed != null) transposed.release();
            left.release();
            right.release();
        }
        assertEquals(0, leftStorage.references());
        assertEquals(0, rightStorage.references());
    }

    @Test
    void scalarUnaryOperationsAndDropoutReadTransposedOffsetViews() {
        HostStorage storage = new HostStorage(99f, 1f, 9f, 4f, 16f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] {1, 2}, 1));
        Tensor powered = null;
        Tensor absolute = null;
        Tensor squareRoot = null;
        Tensor sign = null;
        Tensor dropped = null;
        try {
            assertArrayEquals(new float[] {1f, 4f, 9f, 16f}, input.toFloatArray(), 0f);
            powered = backend.pow(input, 2f);
            absolute = backend.abs(input);
            squareRoot = backend.sqrt(input);
            sign = backend.sign(input);
            dropped = backend.dropout(input, 0.5f, true);

            assertArrayEquals(new float[] {1f, 16f, 81f, 256f}, powered.toFloatArray(), 0f);
            assertArrayEquals(new float[] {1f, 4f, 9f, 16f}, absolute.toFloatArray(), 0f);
            assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, squareRoot.toFloatArray(), 0f);
            assertArrayEquals(new float[] {1f, 1f, 1f, 1f}, sign.toFloatArray(), 0f);
            float[] droppedValues = dropped.toFloatArray();
            float[] sourceValues = {1f, 4f, 9f, 16f};
            for (int index = 0; index < droppedValues.length; index++) {
                assertTrue(droppedValues[index] == 0f
                        || droppedValues[index] == sourceValues[index] * 2f);
            }
        } finally {
            if (dropped != null) dropped.release();
            if (sign != null) sign.release();
            if (squareRoot != null) squareRoot.release();
            if (absolute != null) absolute.release();
            if (powered != null) powered.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void rankTwoAttentionReadsTransposedOffsetQueryKeyAndValueViews() {
        HostStorage queryStorage = new HostStorage(99f, 0f, 0f, 0f, 0f);
        DefaultTensor query = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) queryStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] {1, 2}, 1));
        HostStorage keyStorage = new HostStorage(99f, 1f, 2f, 3f, 4f);
        DefaultTensor key = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) keyStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] {1, 2}, 1));
        HostStorage valueStorage = new HostStorage(99f, 2f, 4f, 3f, 5f);
        DefaultTensor value = new DefaultTensor(
                new Shape(2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) valueStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 2), new long[] {1, 2}, 1));
        Tensor result = null;
        try {
            result = backend.attention(query, key, value);
            assertArrayEquals(new float[] {3f, 4f, 3f, 4f}, result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            query.release();
            key.release();
            value.release();
        }
        assertEquals(0, queryStorage.references());
        assertEquals(0, keyStorage.references());
        assertEquals(0, valueStorage.references());
    }

    @Test
    void rankFourAttentionReadsOffsetViewsAndPreservesLogicalOutput() {
        HostStorage queryStorage = new HostStorage(99f, 0f, 0f, 0f, 0f);
        DefaultTensor query = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) queryStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] {4, 4, 1, 2}, 1));
        HostStorage keyStorage = new HostStorage(99f, 1f, 2f, 3f, 4f);
        DefaultTensor key = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) keyStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] {4, 4, 1, 2}, 1));
        HostStorage valueStorage = new HostStorage(99f, 2f, 4f, 3f, 5f);
        DefaultTensor value = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) valueStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] {4, 4, 1, 2}, 1));
        Tensor result = null;
        try {
            result = backend.attention(query, key, value);
            assertArrayEquals(new float[] {3f, 4f, 3f, 4f}, result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            query.release();
            key.release();
            value.release();
        }
        assertEquals(0, queryStorage.references());
        assertEquals(0, keyStorage.references());
        assertEquals(0, valueStorage.references());
    }


    @Test
    void groupedQueryAttentionReadsStridedOffsetViews() {
        HostStorage queryStorage = new HostStorage(99f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f);
        DefaultTensor query = new DefaultTensor(
                new Shape(1, 2, 2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) queryStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 2, 2, 2), new long[] {8, 4, 1, 2}, 1));
        HostStorage keyStorage = new HostStorage(99f, 1f, 2f, 3f, 4f);
        DefaultTensor key = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) keyStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] {4, 4, 1, 2}, 1));
        HostStorage valueStorage = new HostStorage(99f, 2f, 4f, 3f, 5f);
        DefaultTensor value = new DefaultTensor(
                new Shape(1, 1, 2, 2), DType.F32, DeviceType.CPU,
                (TensorStorage) valueStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(1, 1, 2, 2), new long[] {4, 4, 1, 2}, 1));
        Tensor result = null;
        try {
            result = backend.attentionGqa(query, key, value, null, false);
            assertArrayEquals(new float[] {
                    3f, 4f, 3f, 4f,
                    3f, 4f, 3f, 4f
            }, result.toFloatArray(), 1e-6f);
        } finally {
            if (result != null) result.release();
            query.release();
            key.release();
            value.release();
        }
        assertEquals(0, queryStorage.references());
        assertEquals(0, keyStorage.references());
        assertEquals(0, valueStorage.references());
    }



    @Test
    void splitPreservesOffsetStridesAndSharesStorageAcrossChunks() {
        HostStorage storage = new HostStorage(
                99f, 1f, 2f, 99f, 3f, 4f, 99f, 5f, 6f, 99f, 7f, 8f);
        DefaultTensor input = new DefaultTensor(
                new Shape(4, 2), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(4, 2), new long[] {3, 1}, 1));
        List<Tensor> chunks = List.of();
        try {
            chunks = backend.split(input, 0, 2);
            assertEquals(2, chunks.size());
            assertArrayEquals(new long[] {2, 2}, chunks.get(0).shape().dims());
            assertArrayEquals(new long[] {2, 2}, chunks.get(1).shape().dims());
            assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, chunks.get(0).toFloatArray(), 0f);
            assertArrayEquals(new float[] {5f, 6f, 7f, 8f}, chunks.get(1).toFloatArray(), 0f);
            assertEquals(3, storage.references());
        } finally {
            for (Tensor chunk : chunks) chunk.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }

    @Test
    void splitNormalizesNegativeAxisAndPreservesNonContiguousLogicalOrder() {
        HostStorage storage = new HostStorage(1f, 5f, 2f, 6f, 3f, 7f, 4f, 8f);
        DefaultTensor input = new DefaultTensor(
                new Shape(2, 4), DType.F32, DeviceType.CPU, (TensorStorage) storage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 4), new long[] {1, 2}, 0));
        List<Tensor> chunks = List.of();
        try {
            chunks = backend.split(input, -1, 2);
            assertEquals(2, chunks.size());
            assertArrayEquals(new long[] {2, 2}, chunks.get(0).shape().dims());
            assertArrayEquals(new long[] {2, 2}, chunks.get(1).shape().dims());
            assertArrayEquals(new float[] {1f, 2f, 5f, 6f}, chunks.get(0).toFloatArray(), 0f);
            assertArrayEquals(new float[] {3f, 4f, 7f, 8f}, chunks.get(1).toFloatArray(), 0f);
            assertEquals(3, storage.references());
        } finally {
            for (Tensor chunk : chunks) chunk.release();
            input.release();
        }
        assertEquals(0, storage.references());
    }


    @Test
    void indexSelectReadsOffsetAndStridedSourceAndIndicesInLogicalOrder() {
        HostStorage sourceStorage = new HostStorage(99f, 1f, 4f, 2f, 5f, 3f, 6f);
        HostStorage indexStorage = new HostStorage(99f, 0f, 99f, 0f);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 0, 99);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 4, 2);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 8, 99);
        indexStorage.segment().set(java.lang.foreign.ValueLayout.JAVA_INT, 12, 0);
        DefaultTensor source = new DefaultTensor(
                new Shape(2, 3), DType.F32, DeviceType.CPU, (TensorStorage) sourceStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2, 3), new long[] {1, 2}, 1));
        DefaultTensor indices = new DefaultTensor(
                new Shape(2), DType.I32, DeviceType.CPU, (TensorStorage) indexStorage, backend,
                new tech.kayys.aljabr.core.tensor.TensorLayout(
                        new Shape(2), new long[] {2}, 1));
        Tensor selected = null;
        try {
            selected = backend.indexSelect(source, 1, indices);
            assertArrayEquals(new long[] {2, 2}, selected.shape().dims());
            assertArrayEquals(new float[] {3f, 1f, 6f, 4f}, selected.toFloatArray(), 0f);
        } finally {
            if (selected != null) selected.release();
            indices.release();
            source.release();
        }
        assertEquals(0, sourceStorage.references());
        assertEquals(0, indexStorage.references());
    }

}
