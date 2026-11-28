package tech.kayys.aljabr.cache;

import java.util.Objects;

/**
 * Immutable descriptor for a contiguous region of a borrowed KV allocation.
 *
 * <p>Offsets and lengths are measured in bytes. The descriptor does not grant
 * access by itself; its opaque handle must be consumed by a compatible
 * backend while the originating read view is open.</p>
 *
 * @param handle provider-owned allocation identity
 * @param memoryDomain location of the allocation
 * @param backendId stable backend identifier (for example, cpu, metal, cuda)
 * @param deviceId backend-specific device identity, or empty for host memory
 * @param elementType encoded element representation
 * @param byteOffset offset from the beginning of the allocation
 * @param byteLength length of the region
 */
public record KVCacheBufferRegion(
        KVCacheBufferHandle handle,
        KVCacheMemoryDomain memoryDomain,
        String backendId,
        String deviceId,
        KVCacheElementType elementType,
        long byteOffset,
        long byteLength) {

    public KVCacheBufferRegion {
        Objects.requireNonNull(handle, "handle");
        Objects.requireNonNull(memoryDomain, "memoryDomain");
        Objects.requireNonNull(backendId, "backendId");
        Objects.requireNonNull(deviceId, "deviceId");
        Objects.requireNonNull(elementType, "elementType");
        if (backendId.isBlank()) {
            throw new IllegalArgumentException("backendId must not be blank");
        }
        if (byteOffset < 0 || byteLength < 0) {
            throw new IllegalArgumentException("Region offset and length must not be negative");
        }
        if (byteOffset % elementType.bytes() != 0
                || byteLength % elementType.bytes() != 0) {
            throw new IllegalArgumentException(
                    "Region offset and length must be aligned to the element width");
        }
        if (memoryDomain == KVCacheMemoryDomain.DEVICE && deviceId.isBlank()) {
            throw new IllegalArgumentException("Device regions require a deviceId");
        }
        if (memoryDomain != KVCacheMemoryDomain.DEVICE && !deviceId.isBlank()) {
            throw new IllegalArgumentException("Host regions must not specify a deviceId");
        }
    }

    /** Number of encoded elements in this region. */
    public long elementCount() {
        return byteLength / elementType.bytes();
    }
}
