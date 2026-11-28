package tech.kayys.aljabr.backend.metal.cache;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import tech.kayys.aljabr.cache.KVCacheBufferHandle;

/** Opaque retained MTLBuffer identity owned by MetalKVCacheStorage. */
final class MetalKVCacheBufferHandle implements KVCacheBufferHandle {
    private final MetalKVCacheStorage owner;
    private final MemorySegment objectHandle;
    private final MemorySegment contents;
    private final long byteLength;

    MetalKVCacheBufferHandle(MetalKVCacheStorage owner, MemorySegment objectHandle,
            MemorySegment contents, long byteLength) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.objectHandle = Objects.requireNonNull(objectHandle, "objectHandle");
        this.contents = Objects.requireNonNull(contents, "contents");
        this.byteLength = byteLength;
    }

    <T> T withBuffer(tech.kayys.aljabr.cache.KVCacheBufferReadView lease,
            tech.kayys.aljabr.cache.KVCacheBufferRegion region,
            java.util.function.Function<MetalKVCacheBufferAccess.View, T> consumer) {
        return owner.withMetalBuffer(lease, region, this, consumer);
    }

    MemorySegment objectHandle() { return objectHandle; }
    MemorySegment contents() { return contents; }
    long byteLength() { return byteLength; }
}
