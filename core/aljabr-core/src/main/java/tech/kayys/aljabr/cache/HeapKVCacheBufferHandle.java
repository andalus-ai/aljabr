package tech.kayys.aljabr.cache;

import java.nio.FloatBuffer;
import java.util.Objects;
import java.util.function.Function;

/**
 * Opaque identity for one heap-backed KV array.
 *
 * <p>Only CPU-side code in this package may resolve the backing array. The
 * public cache contract exposes this object only as a {@link KVCacheBufferHandle}.</p>
 */
final class HeapKVCacheBufferHandle implements KVCacheBufferHandle {
    private final HeapKVCacheStorage owner;
    private final float[] storage;

    HeapKVCacheBufferHandle(HeapKVCacheStorage owner, float[] storage) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.storage = Objects.requireNonNull(storage, "storage");
    }

    <T> T withReadOnlyBuffer(
            KVCacheBufferReadView lease,
            KVCacheBufferRegion region,
            Function<FloatBuffer, T> consumer) {
        return owner.withReadOnlyBuffer(lease, region, this, consumer);
    }

    float[] storage() {
        return storage;
    }
}
