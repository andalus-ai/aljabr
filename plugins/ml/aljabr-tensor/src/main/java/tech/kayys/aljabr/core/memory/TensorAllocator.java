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

    /**
     * Allocates storage with the requested byte alignment.
     *
     * <p>The default implementation preserves compatibility with existing
     * allocators that only support unaligned allocations. Allocators that
     * support aligned storage should override this method rather than silently
     * ignoring the requested alignment.</p>
     *
     * @param sizeBytes required capacity in bytes; must be non-negative
     * @param alignmentBytes required alignment in bytes; must be a positive power of two
     * @return a live storage instance with one ownership reference
     * @throws UnsupportedOperationException if this allocator does not support
     *         the requested alignment
     */
    default TensorStorage allocate(long sizeBytes, long alignmentBytes) {
        validateAllocation(sizeBytes, alignmentBytes);
        if (alignmentBytes != 1L) {
            throw new UnsupportedOperationException(
                    "This tensor allocator does not support aligned allocation");
        }
        return allocate(sizeBytes);
    }

    private static void validateAllocation(long sizeBytes, long alignmentBytes) {
        if (sizeBytes < 0L) {
            throw new IllegalArgumentException("sizeBytes must be non-negative");
        }
        if (alignmentBytes <= 0L || (alignmentBytes & (alignmentBytes - 1L)) != 0L) {
            throw new IllegalArgumentException("alignmentBytes must be a positive power of two");
        }
    }
}
