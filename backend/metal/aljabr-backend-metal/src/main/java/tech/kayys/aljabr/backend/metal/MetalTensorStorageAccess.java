package tech.kayys.aljabr.backend.metal;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Storage boundary used by the current FFM-based Metal binding.
 *
 * <p>The native binding accepts host-addressable {@link MemorySegment}
 * instances. This helper makes that requirement explicit without depending
 * on {@code DefaultTensor} or the legacy {@code Buffer} interface.</p>
 */
final class MetalTensorStorageAccess {

    private MetalTensorStorageAccess() {
    }

    static HostAccessibleStorage hostStorage(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor must not be null");
        TensorStorage storage = Objects.requireNonNull(
                tensor.storage(), "tensor.storage() must not return null");
        if (storage instanceof HostAccessibleStorage hostStorage) {
            return hostStorage;
        }
        throw new UnsupportedOperationException(
                "The current Metal FFM binding requires host-accessible tensor storage; tensor="
                        + tensor.getClass().getName() + ", storage="
                        + storage.getClass().getName()
                        + ". Device-local storage requires a Metal-specific binding and allocator.");
    }

    static MemorySegment segment(Tensor tensor) {
        return hostStorage(tensor).segment();
    }
}
