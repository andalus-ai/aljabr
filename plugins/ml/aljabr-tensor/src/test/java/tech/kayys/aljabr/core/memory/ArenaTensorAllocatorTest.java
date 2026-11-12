package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

class ArenaTensorAllocatorTest {

    private final ArenaTensorAllocator allocator = ArenaTensorAllocator.INSTANCE;

    @Test
    void allocatesHostAccessibleStorageWithRequestedCapacity() {
        ArenaTensorStorage storage = allocator.allocate(257L, 64L);

        assertEquals(257L, storage.sizeBytes());
        assertEquals(257L, storage.segment().byteSize());
        assertEquals(0L, storage.segment().address() % 64L);

        storage.release();
    }

    @Test
    void retainsNativeMemoryUntilFinalOwnerReleasesIt() {
        ArenaTensorStorage storage = allocator.allocate(128L, 64L);
        MemorySegment segment = storage.segment();

        storage.retain();
        assertEquals(2, storage.referenceCount());

        storage.release();
        assertTrue(segment.scope().isAlive());
        assertEquals(1, storage.referenceCount());

        storage.release();
        assertFalse(segment.scope().isAlive());
        assertEquals(0, storage.referenceCount());
    }

    @Test
    void rejectsAccessAndOverReleaseAfterFinalRelease() {
        ArenaTensorStorage storage = allocator.allocate(32L);
        storage.release();

        assertThrows(IllegalStateException.class, storage::segment);
        assertThrows(IllegalStateException.class, storage::sizeBytes);
        assertThrows(IllegalStateException.class, storage::retain);
        assertThrows(IllegalStateException.class, storage::release);
    }

    @Test
    void validatesSizeAndAlignmentBeforeAllocating() {
        assertThrows(IllegalArgumentException.class, () -> allocator.allocate(-1L));
        assertThrows(IllegalArgumentException.class, () -> allocator.allocate(8L, 0L));
        assertThrows(IllegalArgumentException.class, () -> allocator.allocate(8L, 3L));
    }

    @Test
    void legacyAllocatorCanStillImplementSingleArgumentContract() {
        TensorAllocator legacy = size -> new TestStorage(size);

        assertEquals(17L, legacy.allocate(17L).sizeBytes());
        assertThrows(UnsupportedOperationException.class, () -> legacy.allocate(17L, 8L));
    }

    private record TestStorage(long sizeBytes) implements TensorStorage {
        @Override
        public void retain() {
        }

        @Override
        public void release() {
        }
    }
}
