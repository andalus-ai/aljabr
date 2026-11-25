package tech.kayys.aljabr.core.memory;

import java.util.List;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.testing.TensorStorageContract;

/**
 * Applies the shared ownership contract to the tensor module's arena storage
 * and retained host-storage view implementations.
 */
class TensorStorageOwnershipContractTest extends TensorStorageContract {

    private final ArenaTensorAllocator allocator = ArenaTensorAllocator.INSTANCE;

    @Override
    protected List<? extends Supplier<? extends TensorStorage>> storageFactories() {
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
