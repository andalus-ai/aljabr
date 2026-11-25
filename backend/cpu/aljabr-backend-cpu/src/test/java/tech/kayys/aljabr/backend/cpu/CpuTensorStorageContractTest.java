package tech.kayys.aljabr.backend.cpu;

import java.util.List;
import java.util.function.Supplier;

import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.testing.TensorStorageContract;

/** Verifies that CPU provider allocations obey the shared storage lifecycle contract. */
class CpuTensorStorageContractTest extends TensorStorageContract {

    @Override
    protected List<? extends Supplier<? extends TensorStorage>> storageFactories() {
        return List.of(() -> new CpuBackend().allocateStorage(64L));
    }
}
