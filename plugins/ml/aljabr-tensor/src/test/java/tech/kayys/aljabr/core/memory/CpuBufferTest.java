package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

class CpuBufferTest {

    @Test
    void cpuBufferImplementsNeutralStorageAndHostAccessContracts() {
        CpuBuffer buffer = new CpuBuffer(16);
        try {
            TensorStorage storage = buffer;
            HostAccessibleStorage hostStorage = buffer;
            Buffer legacyBuffer = buffer;

            assertEquals(16, storage.sizeBytes());
            assertEquals(16, hostStorage.segment().byteSize());
            assertEquals(16, legacyBuffer.sizeBytes());
        } finally {
            buffer.release();
        }
    }

    @Test
    void viewUsesRequestedSliceAndRetainsParentUntilReleased() {
        CpuBuffer parent = new CpuBuffer(128);
        long parentAddress = parent.segment().address();

        CpuBuffer view = CpuBuffer.view(parent, 16, 32);
        assertEquals(32, view.sizeBytes());
        assertEquals(parentAddress + 16, view.segment().address());

        parent.release();
        assertTrue(view.segment().scope().isAlive());

        MemorySegment viewSegment = view.segment();
        view.release();
        assertTrue(!viewSegment.scope().isAlive());
    }

    @Test
    void viewRejectsNegativeAndOutOfBoundsRanges() {
        CpuBuffer parent = new CpuBuffer(64);
        try {
            assertThrows(IllegalArgumentException.class, () -> CpuBuffer.view(parent, -1, 8));
            assertThrows(IllegalArgumentException.class, () -> CpuBuffer.view(parent, 0, -1));
            assertThrows(IllegalArgumentException.class, () -> CpuBuffer.view(parent, 60, 8));
            assertThrows(IllegalArgumentException.class,
                    () -> CpuBuffer.view(parent, Long.MAX_VALUE, 8));
        } finally {
            parent.release();
        }
    }

    @Test
    void legacyViewRejectsUnrelatedNativeSegment() {
        CpuBuffer parent = new CpuBuffer(64);
        try (Arena otherArena = Arena.ofShared()) {
            MemorySegment unrelated = otherArena.allocate(8);
            assertThrows(IllegalArgumentException.class, () -> CpuBuffer.view(unrelated, parent));
        } finally {
            parent.release();
        }
    }

    @Test
    void pooledAllocationIsNotReusedUntilTheLastViewIsReleased() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            MemorySegment firstLease = pool.acquire(64);
            CpuBuffer parent = new CpuBuffer(firstLease, pool.arena(), pool);
            long firstAddress = firstLease.address();
            CpuBuffer view = CpuBuffer.view(parent, 0, 32);

            parent.release();
            MemorySegment secondLease = pool.acquire(64);
            assertTrue(secondLease.address() != firstAddress);

            pool.release(secondLease);
            view.release();

            MemorySegment reusedLease = pool.acquire(64);
            assertEquals(firstAddress, reusedLease.address());
            pool.release(reusedLease);
        }
    }

    @Test
    void releasedBufferRejectsSegmentAccessAfterItsPoolLeaseIsReused() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            MemorySegment lease = pool.acquire(64);
            CpuBuffer buffer = new CpuBuffer(lease, pool.arena(), pool);
            long address = lease.address();

            buffer.release();
            MemorySegment reused = pool.acquire(64);
            assertEquals(address, reused.address());
            assertThrows(IllegalStateException.class, buffer::segment);
            pool.release(reused);
        }
    }

    @Test
    void releasedBufferCannotBeRetainedOrReleasedAgain() {
        CpuBuffer buffer = new CpuBuffer(8);
        buffer.release();
        assertThrows(IllegalStateException.class, buffer::retain);
        assertThrows(IllegalStateException.class, buffer::release);
    }
}
