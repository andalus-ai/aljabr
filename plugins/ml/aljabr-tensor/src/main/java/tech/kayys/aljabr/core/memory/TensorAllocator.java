package tech.kayys.aljabr.core.memory;

/**
 * Allocation policy for tensor-owned storage.
 *
 * <p>Allocators decide where and how storage is obtained (for example pooled
 * host memory, pinned host memory, mapped memory, or device-local memory).
 * Compute backends remain responsible for operations and data transfers.
 * Each successful allocation transfers one ownership reference to its caller.</p>
 */
@FunctionalInterface
public interface TensorAllocator {

    /**
     * Allocates storage with at least {@code sizeBytes} capacity.
     *
     * @param sizeBytes required capacity in bytes; must be non-negative
     * @return a live storage instance with one ownership reference
     */
    TensorStorage allocate(long sizeBytes);
}
