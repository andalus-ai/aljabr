package tech.kayys.aljabr.testing;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.TensorStorage;

/**
 * Reusable lifecycle contract for provider-owned tensor storage.
 *
 * <p>Backend modules can subclass this suite and supply their storage
 * allocation factories. The tests intentionally assert only the public
 * {@link TensorStorage} contract, not implementation details.</p>
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

            // The interface defines final release as the end of the lifetime,
            // but deliberately does not prescribe post-release exception behavior.
            storage.release();
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

            // Do not inspect the storage after relinquishing the final owner.
            storage.release();
        }
    }
}
