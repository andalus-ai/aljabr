package tech.kayys.aljabr.testing;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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

    /**
     * Returns factories for storage implementations owned by the provider
     * under test. Each invocation must return a fresh, independently-owned
     * storage instance with at least the requested capacity.
     */
    protected abstract List<? extends Supplier<? extends TensorStorage>> storageFactories();

    @Test
    public final void implementationsHonorBalancedOwnershipAndFinalRelease() {
        for (Supplier<? extends TensorStorage> factory : storageFactories()) {
            TensorStorage storage = factory.get();
            long capacity = storage.sizeBytes();

            assertEquals(64L, capacity);
            storage.retain();
            assertEquals(capacity, storage.sizeBytes());

            storage.release();
            assertEquals(capacity, storage.sizeBytes());

            storage.release();
            assertThrows(IllegalStateException.class, storage::sizeBytes);
            assertThrows(IllegalStateException.class, storage::retain);
            assertThrows(IllegalStateException.class, storage::release);
        }
    }

    @Test
    public final void multipleRetainsRequireTheSameNumberOfReleases() {
        for (Supplier<? extends TensorStorage> factory : storageFactories()) {
            TensorStorage storage = factory.get();

            storage.retain();
            storage.retain();
            storage.release();
            assertEquals(64L, storage.sizeBytes());

            storage.release();
            assertEquals(64L, storage.sizeBytes());

            storage.release();
            assertThrows(IllegalStateException.class, storage::sizeBytes);
        }
    }
}
