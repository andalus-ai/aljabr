package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class ManagedArenaTest {

    @Test
    void closesUnderlyingArenaOnlyAfterFinalReference() {
        ManagedArena managed = ManagedArena.create();
        var segment = managed.allocate(16);
        managed.retain();

        managed.close();
        assertTrue(managed.isOpen());
        assertTrue(segment.scope().isAlive());
        assertEquals(1, managed.refCount());

        managed.close();
        assertFalse(managed.isOpen());
        assertFalse(segment.scope().isAlive());
        assertEquals(0, managed.refCount());
    }

    @Test
    void rejectsOverReleaseWithoutCorruptingReferenceCount() {
        ManagedArena managed = ManagedArena.create();
        managed.close();

        assertThrows(IllegalStateException.class, managed::close);
        assertEquals(0, managed.refCount());
        assertFalse(managed.isOpen());
    }

    @Test
    void rejectsRetainAfterFinalClose() {
        ManagedArena managed = ManagedArena.create();
        managed.close();

        assertThrows(IllegalStateException.class, managed::retain);
        assertEquals(0, managed.refCount());
    }

    @Test
    void rejectsNullArena() {
        assertThrows(NullPointerException.class, () -> ManagedArena.wrap(null));
    }

    @Test
    void concurrentRetainsAndReleasesPreserveOwnerReference() throws Exception {
        ManagedArena managed = ManagedArena.create();
        int workers = 8;
        var pool = java.util.concurrent.Executors.newFixedThreadPool(workers);
        try {
            var tasks = java.util.stream.IntStream.range(0, workers)
                    .<java.util.concurrent.Callable<Void>>mapToObj(i -> () -> {
                        managed.retain();
                        try {
                            assertTrue(managed.isOpen());
                        } finally {
                            managed.close();
                        }
                        return null;
                    }).toList();
            for (var future : pool.invokeAll(tasks)) {
                future.get();
            }
            assertEquals(1, managed.refCount());
        } finally {
            pool.shutdownNow();
            managed.close();
        }
        assertEquals(0, managed.refCount());
    }
}
