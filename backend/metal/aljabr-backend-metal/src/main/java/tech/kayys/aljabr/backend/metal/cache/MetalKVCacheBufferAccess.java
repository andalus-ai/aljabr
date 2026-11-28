package tech.kayys.aljabr.backend.metal.cache;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Function;

import tech.kayys.aljabr.cache.KVCacheBufferReadView;
import tech.kayys.aljabr.cache.KVCacheBufferRegion;
import tech.kayys.aljabr.cache.KVCacheElementType;
import tech.kayys.aljabr.cache.KVCacheMemoryDomain;

/**
 * Scoped access to a borrowed Metal KV buffer.
 *
 * <p>The MTLBuffer reference and CPU-visible contents slice are valid only
 * during the callback and while the originating lease remains open. Do not
 * retain either value. The native handle is an Objective-C object pointer,
 * not a dereferenceable memory address.</p>
 */
public final class MetalKVCacheBufferAccess {
    private MetalKVCacheBufferAccess() { }

    public record View(MemorySegment nativeBuffer, MemorySegment readOnlyContents,
            long byteLength, String deviceId) { }

    public static <T> T withBuffer(KVCacheBufferReadView lease,
            KVCacheBufferRegion region, Function<View, T> consumer) {
        Objects.requireNonNull(lease, "lease");
        Objects.requireNonNull(region, "region");
        Objects.requireNonNull(consumer, "consumer");
        if (region.memoryDomain() != KVCacheMemoryDomain.DEVICE
                || !"metal".equals(region.backendId())
                || region.elementType() != KVCacheElementType.FLOAT32
                || !(region.handle() instanceof MetalKVCacheBufferHandle handle)) {
            throw new IllegalArgumentException("Region is not a Metal DEVICE FLOAT32 allocation");
        }
        return handle.withBuffer(lease, region, consumer);
    }
}
