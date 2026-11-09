package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

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
    void contiguousSliceIsZeroCopyAndRetainsNonBufferParent() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f);
        DefaultTensor input = tensor(storage, 2, 2);
        Tensor slice = backend.slice(input, new long[] { 1, 0 }, new long[] { 1, 2 });
        try {
            assertNotSame(storage, slice.storage());
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
    void nonContiguousSliceMaterializesIndependentCpuStorage() {
        HostStorage storage = new HostStorage(1f, 2f, 3f, 4f, 5f, 6f);
        DefaultTensor input = tensor(storage, 2, 3);
        Tensor slice = null;
        try {
            slice = backend.slice(input, new long[] { 0, 1 }, new long[] { 2, 2 });
            assertNotSame(storage, slice.storage());
            assertArrayEquals(new float[] { 2f, 3f, 5f, 6f }, slice.toFloatArray(), 0f);
            assertEquals(1, storage.references());
        } finally {
            if (slice != null) {
                slice.release();
            }
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

}
