package tech.kayys.aljabr.cache;

import java.util.Objects;

/**
 * Opaque identity for one heap-backed KV array.
 *
 * <p>Only CPU-side code in this package may resolve the backing array. The
 * public cache contract exposes this object only as a {@link KVCacheBufferHandle}.</p>
 */
final class HeapKVCacheBufferHandle implements KVCacheBufferHandle {
    private final float[] storage;

    HeapKVCacheBufferHandle(float[] storage) {
        this.storage = Objects.requireNonNull(storage, "storage");
    }

    float[] storage() {
        return storage;
    }
}
