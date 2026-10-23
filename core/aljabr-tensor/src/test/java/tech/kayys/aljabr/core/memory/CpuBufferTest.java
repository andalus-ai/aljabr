package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

class CpuBufferTest {

    @Test
    void viewUsesRequestedSliceAndRetainsParentUntilReleased() {
        CpuBuffer parent = new CpuBuffer(128);
        long parentAddress = parent.segment().address();

        CpuBuffer view = CpuBuffer.view(parent, 16, 32);
        assertEquals(32, view.sizeBytes());
        assertEquals(parentAddress + 16, view.segment().address());

        parent.release();
        assertTrue(view.segment().scope().isAlive());

        view.release();
        assertTrue(!view.segment().scope().isAlive());
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
    void releasedBufferCannotBeRetainedOrReleasedAgain() {
        CpuBuffer buffer = new CpuBuffer(8);
        buffer.release();
        assertThrows(IllegalStateException.class, buffer::retain);
        assertThrows(IllegalStateException.class, buffer::release);
    }
}
