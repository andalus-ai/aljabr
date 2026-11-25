package tech.kayys.aljabr.testing;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Future;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;

/**
 * Reusable lifecycle contract for provider-owned tensor storage.
 *
 * <p>Backend modules can subclass this suite and supply their storage
 * allocation factories. The tests assert only the public {@link TensorStorage}
 * ownership contract and optional host-accessibility capability.</p>
 */
public abstract class TensorStorageContract {

    private static final long MINIMUM_CAPACITY_BYTES = 64L;

    /**
     * Returns factories for storage implementations owned by the provider
     * under test. Each invocation must return a fresh, independently-owned
     * storage instance with at least the minimum requested capacity.
     */
    protected abstract List<? extends Supplier<? extends TensorStorage>> storageFactories();

    @Test
    public final void implementationsHonorBalancedOwnershipAndFinalRelease() {
        for (Supplier<? extends TensorStorage> factory : storageFactories()) {
            TensorStorage storage = factory.get();
            long capacity = storage.sizeBytes();

            assertTrue(capacity >= MINIMUM_CAPACITY_BYTES,
                    "storage must provide at least " + MINIMUM_CAPACITY_BYTES + " bytes");
            storage.retain();
            assertEquals(capacity, storage.sizeBytes());

            storage.release();
            assertEquals(capacity, storage.sizeBytes());

            storage.release();
            assertReleased(storage, "final release must end storage lifetime");
        }
    }

    @Test
    public final void multipleRetainsRequireTheSameNumberOfReleases() {
        for (Supplier<? extends TensorStorage> factory : storageFactories()) {
            TensorStorage storage = factory.get();
            long capacity = storage.sizeBytes();

            storage.retain();
            storage.retain();
            storage.release();
            assertEquals(capacity, storage.sizeBytes());

            storage.release();
            assertEquals(capacity, storage.sizeBytes());

            storage.release();
            assertReleased(storage, "all acquired references must be balanced");
        }
    }

    @Test
    public final void finalReleaseRejectsFurtherOwnershipAndAccess() {
        for (Supplier<? extends TensorStorage> factory : storageFactories()) {
            TensorStorage storage = factory.get();
            storage.release();

            assertReleased(storage, "storage must reject use after final release");
            assertThrows(IllegalStateException.class, storage::retain);
            assertThrows(IllegalStateException.class, storage::release);
        }
    }

    @Test
    public final void retainRacingWithFinalReleaseNeverResurrectsStorage() throws Exception {
        try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
            for (Supplier<? extends TensorStorage> factory : storageFactories()) {
                for (int iteration = 0; iteration < 32; iteration++) {
                    TensorStorage storage = factory.get();
                    MemorySegment segment = storage instanceof HostAccessibleStorage host
                            ? host.segment()
                            : null;
                    CountDownLatch ready = new CountDownLatch(1);
                    CountDownLatch start = new CountDownLatch(1);

                    Future<?> retainedRelease = executor.submit(() -> {
                        ready.countDown();
                        if (!start.await(2, TimeUnit.SECONDS)) {
                            throw new AssertionError("timed out waiting for race start");
                        }
                        boolean retained;
                        try {
                            storage.retain();
                            retained = true;
                        } catch (IllegalStateException finalReleaseWon) {
                            retained = false;
                        }
                        if (retained) {
                            storage.release();
                        }
                        return null;
                    });

                    assertTrue(ready.await(2, TimeUnit.SECONDS),
                            "retainer must be ready before final release");
                    start.countDown();
                    storage.release();
                    retainedRelease.get(2, TimeUnit.SECONDS);

                    assertReleased(storage,
                            "storage must be closed after retain/final-release race " + iteration);
                    if (segment != null) {
                        assertFalse(segment.scope().isAlive(),
                                "host segment must be invalidated after final release");
                    }
                }
            }
        }
    }

    private static void assertReleased(TensorStorage storage, String message) {
        assertThrows(IllegalStateException.class, storage::sizeBytes, message);
        if (storage instanceof HostAccessibleStorage host) {
            assertThrows(IllegalStateException.class, host::segment, message);
        }
    }
}
