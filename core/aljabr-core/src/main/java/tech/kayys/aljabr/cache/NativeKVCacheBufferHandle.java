package tech.kayys.aljabr.cache;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Function;

/** Opaque identity for one FFM-owned native KV allocation. */
final class NativeKVCacheBufferHandle implements KVCacheBufferHandle {
    private final NativeKVCacheStorage owner;
    private final MemorySegment segment;

    NativeKVCacheBufferHandle(NativeKVCacheStorage owner, MemorySegment segment) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.segment = Objects.requireNonNull(segment, "segment");
    }

    <T> T withReadOnlySegment(KVCacheBufferReadView lease, KVCacheBufferRegion region,
            Function<MemorySegment, T> consumer) {
        return owner.withReadOnlySegment(lease, region, this, consumer);
    }

    MemorySegment segment() { return segment; }
}
