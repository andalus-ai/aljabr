package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Shared host-storage boundary for CPU kernels.
 *
 * <p>CPU operations work against the host-accessible capability rather than a
 * particular eager tensor implementation or the legacy Buffer type.</p>
 */
final class CpuTensorStorageAccess {

    private CpuTensorStorageAccess() {
    }

    static HostAccessibleStorage hostStorage(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor must not be null");
        TensorStorage storage = Objects.requireNonNull(
                tensor.storage(), "tensor.storage() must not return null");
        if (storage instanceof HostAccessibleStorage hostStorage) {
            return hostStorage;
        }
        throw new UnsupportedOperationException(
                "CPU execution requires host-accessible tensor storage; tensor="
                        + tensor.getClass().getName() + ", storage="
                        + storage.getClass().getName());
    }

    static MemorySegment segment(Tensor tensor) {
        return hostStorage(tensor).segment();
    }
}
