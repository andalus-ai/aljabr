package tech.kayys.aljabr.cache;

import java.nio.FloatBuffer;
import java.util.Objects;
import java.util.function.Function;

/**
 * Scoped read access for CPU heap-backed KV regions.
 *
 * <p>The callback receives a read-only, zero-copy {@link FloatBuffer} view.
 * The view is valid only for the duration of the callback; consumers must not
 * retain it or use it asynchronously. The lease remains responsible for
 * protecting the allocation and its visible prefix.</p>
 */
public final class HeapKVCacheBufferAccess {

    private HeapKVCacheBufferAccess() {
    }

    /**
     * Applies a synchronous CPU read operation to a borrowed heap region.
     *
     * @param lease active lease that exported the region
     * @param region K or V region from that lease
     * @param consumer operation that reads the region without retaining the buffer
     * @param <T> callback result
     * @return callback result
     * @throws IllegalArgumentException if the region is not a CPU heap FLOAT32 region
     * @throws IllegalStateException if the lease is closed or does not own the region
     */
    public static <T> T withReadOnlyBuffer(
            KVCacheBufferReadView lease,
            KVCacheBufferRegion region,
            Function<FloatBuffer, T> consumer) {
        Objects.requireNonNull(lease, "lease");
        Objects.requireNonNull(region, "region");
        Objects.requireNonNull(consumer, "consumer");
        if (region.memoryDomain() != KVCacheMemoryDomain.HOST_HEAP
                || !"cpu".equals(region.backendId())
                || region.elementType() != KVCacheElementType.FLOAT32
                || !(region.handle() instanceof HeapKVCacheBufferHandle handle)) {
            throw new IllegalArgumentException("Region is not a CPU heap FLOAT32 allocation");
        }
        return handle.withReadOnlyBuffer(lease, region, consumer);
    }
}
