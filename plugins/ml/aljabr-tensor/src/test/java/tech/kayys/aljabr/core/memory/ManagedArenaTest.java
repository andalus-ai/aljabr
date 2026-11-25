package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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
    void retainRacingWithFinalCloseNeverResurrectsOrOverReleasesArena() throws Exception {
        try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
            for (int iteration = 0; iteration < 64; iteration++) {
                ManagedArena managed = ManagedArena.ofShared();
                var segment = managed.allocate(8);
                CountDownLatch ready = new CountDownLatch(2);
                CountDownLatch start = new CountDownLatch(1);

                var retainAndRelease = executor.submit(() -> {
                    ready.countDown();
                    if (!start.await(2, TimeUnit.SECONDS)) {
                        throw new AssertionError("timed out waiting for race start");
                    }
                    boolean retained;
                    try {
                        managed.retain();
                        retained = true;
                    } catch (IllegalStateException alreadyClosed) {
                        // The competing final close linearized first.
                        retained = false;
                    }
                    if (retained) {
                        managed.close();
                    }
                    return null;
                });
                var finalRelease = executor.submit(() -> {
                    ready.countDown();
                    if (!start.await(2, TimeUnit.SECONDS)) {
                        throw new AssertionError("timed out waiting for race start");
                    }
                    managed.close();
                    return null;
                });

                assertTrue(ready.await(2, TimeUnit.SECONDS));
                start.countDown();
                retainAndRelease.get(2, TimeUnit.SECONDS);
                finalRelease.get(2, TimeUnit.SECONDS);

                assertEquals(0, managed.refCount(), "reference count at iteration " + iteration);
                assertFalse(managed.isOpen(), "arena must be closed at iteration " + iteration);
                assertFalse(segment.scope().isAlive(), "segment must be invalidated at iteration " + iteration);
            }
        }
    }

    @Test
    void rejectsNullArena() {
        assertThrows(NullPointerException.class, () -> new ManagedArena(null));
    }

    @Test
    void preservesCoreCompatibilityFactoriesAndRawAccessor() {
        ManagedArena created = ManagedArena.create();
        assertSame(created.arena(), created.raw());
        created.close();

        Arena wrappedArena = Arena.ofShared();
        ManagedArena wrapped = ManagedArena.wrap(wrappedArena);
        assertSame(wrappedArena, wrapped.raw());
        wrapped.close();
        assertFalse(wrappedArena.scope().isAlive());
    }

    @Test
    void reportsExternallyClosedWrappedArenaAsNotOpen() {
        Arena external = Arena.ofShared();
        ManagedArena managed = ManagedArena.wrap(external);
        external.close();

        assertFalse(managed.isOpen());
        assertThrows(IllegalStateException.class, () -> managed.allocate(1));
        // Release the wrapper's reference without attempting to close the arena twice.
        managed.close();
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
