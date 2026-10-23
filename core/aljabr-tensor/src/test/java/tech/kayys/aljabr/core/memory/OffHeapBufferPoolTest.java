package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;

class OffHeapBufferPoolTest {
    @Test
    void finalBufferReleaseReturnsSegmentToPoolAndUpdatesTelemetry() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            CpuBuffer first = new CpuBuffer(pool.acquire(Float.BYTES), pool.arena(), pool);
            assertEquals(0, pool.bytesPooled());
            first.release();
            assertEquals(64, pool.bytesPooled());

            CpuBuffer second = new CpuBuffer(pool.acquire(Float.BYTES), pool.arena(), pool);
            assertEquals(0, pool.bytesPooled());
            assertEquals(1, pool.hits());
            second.release();
            assertEquals(64, pool.bytesPooled());
        }
    }

    @Test
    void tensorArrayExportUsesLogicalShapeNotPoolBucketCapacity() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            CpuBuffer buffer = new CpuBuffer(pool.acquire(Float.BYTES), pool.arena(), pool);
            buffer.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 2.25f);
            DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, null);
            try {
                assertArrayEquals(new float[] {2.25f}, tensor.toFloatArray(), 0f);
            } finally {
                tensor.release();
            }
            assertEquals(64, pool.bytesPooled());
        }
    }

    @Test
    void closeIsIdempotentAndPreventsNewAllocations() {
        OffHeapBufferPool pool = new OffHeapBufferPool();
        pool.close();
        assertDoesNotThrow(pool::close);
        assertThrows(IllegalStateException.class, () -> pool.acquire(Float.BYTES));
    }

    @Test
    void rejectsDoubleRelease() {
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        buffer.release();
        assertThrows(IllegalStateException.class, buffer::release);
    }
}
