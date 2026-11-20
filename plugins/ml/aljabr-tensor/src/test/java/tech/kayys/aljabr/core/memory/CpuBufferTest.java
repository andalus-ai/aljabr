package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

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
    void siblingViewsObserveSharedWritesAndOwnIndependentParentReferences() {
        CpuBuffer parent = new CpuBuffer(32);
        CpuBuffer first = CpuBuffer.view(parent, 8, 8);
        CpuBuffer second = CpuBuffer.view(parent, 8, 8);
        MemorySegment secondSegment = second.segment();

        try {
            first.segment().set(ValueLayout.JAVA_INT, 0, 0x12345678);
            assertEquals(0x12345678, secondSegment.get(ValueLayout.JAVA_INT, 0));

            parent.release();
            first.release();
            assertTrue(secondSegment.scope().isAlive());
            assertEquals(0x12345678, secondSegment.get(ValueLayout.JAVA_INT, 0));
        } finally {
            second.release();
        }

        assertTrue(!secondSegment.scope().isAlive());
    }

    @Test
    void nestedViewKeepsOriginalAllocationAliveAfterIntermediateViewsRelease() {
        CpuBuffer parent = new CpuBuffer(64);
        CpuBuffer outer = CpuBuffer.view(parent, 16, 32);
        CpuBuffer inner = CpuBuffer.view(outer, 8, 8);
        MemorySegment innerSegment = inner.segment();

        parent.release();
        outer.release();

        try {
            assertTrue(innerSegment.scope().isAlive());
            innerSegment.set(ValueLayout.JAVA_LONG, 0, 42L);
            assertEquals(42L, innerSegment.get(ValueLayout.JAVA_LONG, 0));
        } finally {
            inner.release();
        }

        assertTrue(!innerSegment.scope().isAlive());
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
