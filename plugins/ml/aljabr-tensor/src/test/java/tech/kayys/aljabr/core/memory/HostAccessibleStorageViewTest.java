package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

class HostAccessibleStorageViewTest {

    private final ArenaTensorAllocator allocator = ArenaTensorAllocator.INSTANCE;

    @Test
    void retainsParentUntilFinalViewRelease() {
        ArenaTensorStorage parent = allocator.allocate(64L);
        MemorySegment parentSegment = parent.segment();
        HostAccessibleStorageView view = HostAccessibleStorageView.view(parent, 8L, 16L);
        MemorySegment viewSegment = view.segment();

        assertEquals(2, parent.referenceCount());
        assertEquals(16L, view.sizeBytes());
        assertEquals(parentSegment.address() + 8L, viewSegment.address());

        parent.release();

        assertEquals(1, parent.referenceCount());
        assertTrue(viewSegment.scope().isAlive());
        viewSegment.set(ValueLayout.JAVA_LONG, 0L, 42L);
        assertEquals(42L, viewSegment.get(ValueLayout.JAVA_LONG, 0L));

        view.release();

        assertEquals(0, parent.referenceCount());
        assertFalse(viewSegment.scope().isAlive());
    }

    @Test
    void siblingAndNestedViewsKeepSharedParentStorageAliveIndependently() {
        ArenaTensorStorage parent = allocator.allocate(64L);
        HostAccessibleStorageView first = HostAccessibleStorageView.view(parent, 8L, 24L);
        HostAccessibleStorageView second = HostAccessibleStorageView.view(parent, 8L, 24L);
        HostAccessibleStorageView nested = HostAccessibleStorageView.view(first, 8L, 8L);

        MemorySegment firstSegment = first.segment();
        MemorySegment secondSegment = second.segment();
        MemorySegment nestedSegment = nested.segment();

        firstSegment.set(ValueLayout.JAVA_INT, 0L, 0x12345678);
        assertEquals(0x12345678, secondSegment.get(ValueLayout.JAVA_INT, 0L));
        assertEquals(firstSegment.address() + 8L, nestedSegment.address());

        parent.release();
        first.release();

        assertTrue(secondSegment.scope().isAlive());
        assertTrue(nestedSegment.scope().isAlive());
        assertEquals(0x12345678, secondSegment.get(ValueLayout.JAVA_INT, 0L));

        nested.release();
        assertTrue(secondSegment.scope().isAlive());
        second.release();

        assertFalse(secondSegment.scope().isAlive());
    }

    @Test
    void rejectsAccessAndReferenceOperationsAfterFinalRelease() {
        ArenaTensorStorage parent = allocator.allocate(32L);
        HostAccessibleStorageView view = HostAccessibleStorageView.view(parent, 0L, 16L);
        MemorySegment segment = view.segment();

        view.release();

        assertFalse(segment.scope().isAlive());
        assertThrows(IllegalStateException.class, view::segment);
        assertThrows(IllegalStateException.class, view::sizeBytes);
        assertThrows(IllegalStateException.class, view::retain);
        assertThrows(IllegalStateException.class, view::release);
        assertEquals(0, parent.referenceCount());
    }

    @Test
    void rejectsInvalidRangesWithoutRetainingParent() {
        ArenaTensorStorage parent = allocator.allocate(32L);
        try {
            assertThrows(NullPointerException.class,
                    () -> HostAccessibleStorageView.view(null, 0L, 1L));
            assertThrows(IllegalArgumentException.class,
                    () -> HostAccessibleStorageView.view(parent, -1L, 8L));
            assertThrows(IllegalArgumentException.class,
                    () -> HostAccessibleStorageView.view(parent, 0L, -1L));
            assertThrows(IllegalArgumentException.class,
                    () -> HostAccessibleStorageView.view(parent, 24L, 16L));
            assertThrows(IllegalArgumentException.class,
                    () -> HostAccessibleStorageView.view(parent, Long.MAX_VALUE, 1L));

            assertEquals(1, parent.referenceCount());
            assertEquals(32L, parent.sizeBytes());
        } finally {
            parent.release();
        }
    }
}
