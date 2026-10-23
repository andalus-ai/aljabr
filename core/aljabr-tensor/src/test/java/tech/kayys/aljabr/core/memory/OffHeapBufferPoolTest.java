package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

class OffHeapBufferPoolTest {
    @Test
    void reusesReturnedSegmentsAndTracksPooledBytes() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            MemorySegment first = pool.acquire(80);
            assertEquals(128, first.byteSize());
            long address = first.address();
            pool.release(first);
            assertEquals(128, pool.bytesPooled());
            MemorySegment second = pool.acquire(80);
            assertEquals(address, second.address());
            assertEquals(1, pool.hits());
            assertEquals(0, pool.bytesPooled());
            pool.release(second);
        }
    }

    @Test
    void rejectsNegativeSizesAndSegmentsFromOtherArenas() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool(); Arena other = Arena.ofShared()) {
            assertThrows(IllegalArgumentException.class, () -> pool.acquire(-1));
            MemorySegment foreign = other.allocate(64);
            assertThrows(IllegalArgumentException.class, () -> pool.release(foreign));
        }
    }

    @Test
    void rejectsSegmentsNotCurrentlyLeasedAndDuplicateRelease() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            MemorySegment acquired = pool.acquire(64);
            pool.release(acquired);
            assertThrows(IllegalArgumentException.class, () -> pool.release(acquired));

            // Matching arena scope alone is not proof that this pool issued a lease.
            MemorySegment unleased = pool.arena().allocate(64);
            assertThrows(IllegalArgumentException.class, () -> pool.release(unleased));

            MemorySegment reused = pool.acquire(64);
            assertEquals(acquired.address(), reused.address());
            pool.release(reused);
        }
    }

    @Test
    void closeIsIdempotentAndRejectsSubsequentAcquire() {
        OffHeapBufferPool pool = new OffHeapBufferPool();
        MemorySegment segment = pool.acquire(64);
        pool.release(segment);
        pool.close();
        pool.close();
        assertEquals(0, pool.bytesPooled());
        assertThrows(IllegalStateException.class, () -> pool.acquire(64));
        assertFalse(segment.scope().isAlive());
    }

    @Test
    void concurrentAcquireAndReleaseRemainSafeWhenCloseRaces() throws Exception {
        OffHeapBufferPool pool = new OffHeapBufferPool();
        ExecutorService executor = Executors.newSingleThreadExecutor();
        CountDownLatch firstAcquireCompleted = new CountDownLatch(1);
        try {
            Future<?> worker = executor.submit(() -> {
                for (int i = 0; i < 10_000; i++) {
                    final MemorySegment segment;
                    try {
                        segment = pool.acquire(128);
                    } catch (IllegalStateException closed) {
                        return;
                    }
                    firstAcquireCompleted.countDown();
                    // Do not dereference the segment here: close is allowed to
                    // invalidate leases once callers stop using them.
                    pool.release(segment);
                }
            });
            // Ensure the race includes at least one successful acquire; without
            // this barrier, close could win before the worker exercises the pool.
            assertTrue(firstAcquireCompleted.await(5, TimeUnit.SECONDS));
            pool.close();
            worker.get(5, TimeUnit.SECONDS);
            assertThrows(IllegalStateException.class, () -> pool.acquire(128));
        } finally {
            pool.close();
            executor.shutdownNow();
        }
    }
}
