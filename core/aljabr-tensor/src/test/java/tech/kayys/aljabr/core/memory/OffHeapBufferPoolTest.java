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
            pool.release(first);
            assertEquals(128, pool.bytesPooled());
            MemorySegment second = pool.acquire(80);
            assertEquals(first.address(), second.address());
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
    void closeWaitsForAnInFlightAcquireOperation() throws Exception {
        OffHeapBufferPool pool = new OffHeapBufferPool();
        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch started = new CountDownLatch(1);
        try {
            Future<?> acquire = executor.submit(() -> {
                started.countDown();
                MemorySegment segment = pool.acquire(128);
                assertTrue(segment.scope().isAlive());
                pool.release(segment);
            });
            assertTrue(started.await(5, TimeUnit.SECONDS));
            acquire.get(5, TimeUnit.SECONDS);
            Future<?> close = executor.submit(pool::close);
            close.get(5, TimeUnit.SECONDS);
            assertThrows(IllegalStateException.class, () -> pool.acquire(128));
        } finally {
            pool.close();
            executor.shutdownNow();
        }
    }
}
