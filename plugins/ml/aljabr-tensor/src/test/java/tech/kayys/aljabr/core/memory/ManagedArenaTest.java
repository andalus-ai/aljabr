package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class ManagedArenaTest {

    @Test
    void closesUnderlyingArenaOnlyAfterFinalReference() {
        ManagedArena managed = ManagedArena.ofShared();
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
        ManagedArena managed = ManagedArena.ofShared();
        managed.close();

        assertThrows(IllegalStateException.class, managed::close);
        assertEquals(0, managed.refCount());
        assertFalse(managed.isOpen());
    }

    @Test
    void rejectsRetainAfterFinalClose() {
        ManagedArena managed = ManagedArena.ofShared();
        managed.close();

        assertThrows(IllegalStateException.class, managed::retain);
        assertEquals(0, managed.refCount());
    }

    @Test
    void rejectsNullArena() {
        assertThrows(NullPointerException.class, () -> new ManagedArena(null));
    }

    @Test
    void confinedArenaRemainsConfinedDespiteReferenceCounting() {
        ManagedArena managed = ManagedArena.ofConfined();
        var segment = managed.allocate(8);
        assertTrue(segment.scope().isAlive());
        managed.close();
        assertFalse(segment.scope().isAlive());
    }
}
