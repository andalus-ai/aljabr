package tech.kayys.aljabr.cache;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Function;

/**
 * Scoped read access for CPU native-memory KV regions.
 *
 * <p>The callback receives a read-only segment slice. Do not retain the slice
 * or use it in work that outlives the callback and active lease. Storage
 * mutation and close are serialized with callback execution.</p>
 */
public final class NativeKVCacheBufferAccess {
    private NativeKVCacheBufferAccess() { }

    public static <T> T withReadOnlySegment(KVCacheBufferReadView lease,
            KVCacheBufferRegion region, Function<MemorySegment, T> consumer) {
        Objects.requireNonNull(lease, "lease");
        Objects.requireNonNull(region, "region");
        Objects.requireNonNull(consumer, "consumer");
        if (region.memoryDomain() != KVCacheMemoryDomain.HOST_NATIVE
                || !"cpu".equals(region.backendId())
                || region.elementType() != KVCacheElementType.FLOAT32
                || !(region.handle() instanceof NativeKVCacheBufferHandle handle)) {
            throw new IllegalArgumentException("Region is not a CPU native FLOAT32 allocation");
        }
        return handle.withReadOnlySegment(lease, region, consumer);
    }
}
