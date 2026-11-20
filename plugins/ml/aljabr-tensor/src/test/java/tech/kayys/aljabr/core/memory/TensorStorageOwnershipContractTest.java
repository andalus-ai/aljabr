package tech.kayys.aljabr.core.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

/**
 * Shared ownership contract checks for concrete TensorStorage implementations.
 *
 * <p>Keep these assertions implementation-neutral: provider-specific storage
 * behavior belongs in the implementation's own tests.</p>
 */
class TensorStorageOwnershipContractTest {

    private final ArenaTensorAllocator allocator = ArenaTensorAllocator.INSTANCE;

    @Test
    void implementationsHonorBalancedOwnershipAndFinalRelease() {
        for (Supplier<TensorStorage> factory : storageFactories()) {
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
    void multipleRetainsRequireTheSameNumberOfReleases() {
        for (Supplier<TensorStorage> factory : storageFactories()) {
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

    private List<Supplier<TensorStorage>> storageFactories() {
        return List.of(
                () -> allocator.allocate(64L),
                this::newViewWithTransferredParentOwnership);
    }

    private TensorStorage newViewWithTransferredParentOwnership() {
        ArenaTensorStorage parent = allocator.allocate(64L);
        HostAccessibleStorageView view = HostAccessibleStorageView.view(parent, 0L, 64L);
        // The view has retained its own parent reference; relinquish the
        // factory's initial reference so the view owns the remaining lifetime.
        parent.release();
        return view;
    }
}
