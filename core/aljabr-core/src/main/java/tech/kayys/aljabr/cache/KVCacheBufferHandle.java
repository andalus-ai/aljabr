package tech.kayys.aljabr.cache;

/**
 * Opaque provider-owned identity for a KV cache allocation.
 *
 * <p>This marker intentionally exposes no address, pointer, Java array, or
 * device-specific object. Only a compatible backend/provider may interpret
 * the handle. A handle is valid only while its owning
 * {@link KVCacheBufferReadView} remains open.</p>
 */
public interface KVCacheBufferHandle {
}
