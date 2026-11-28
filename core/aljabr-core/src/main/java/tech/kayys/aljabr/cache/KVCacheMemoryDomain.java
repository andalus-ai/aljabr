package tech.kayys.aljabr.cache;

/**
 * Memory domain in which a borrowed KV cache region resides.
 *
 * <p>The domain is descriptive, not an instruction to dereference memory.
 * Consumers must use a compatible backend/provider to access a region.</p>
 */
public enum KVCacheMemoryDomain {
    /** Java-managed heap memory; direct access is provider-specific. */
    HOST_HEAP,
    /** Native host memory, including memory represented by a foreign-memory segment. */
    HOST_NATIVE,
    /** Memory resident on an accelerator device. */
    DEVICE
}
