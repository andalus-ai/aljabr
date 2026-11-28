package tech.kayys.aljabr.cache;

/**
 * Optional capability for borrowing backend-native KV allocation regions.
 *
 * <p>Implementations advertise this capability only when they can guarantee
 * allocation lifetime, mutation exclusion/coherence, and descriptor validity
 * for the complete lease. The capability is separate from
 * {@link KVCacheSharingCapability}: a provider may support logical copied
 * reads without exposing native/device regions.</p>
 */
public interface KVCacheBufferSharingCapability {

    /**
     * Opens a backend-aware read lease bounded to the currently committed
     * sequence prefix.
     *
     * @return a caller-closeable borrowed buffer view
     */
    KVCacheBufferReadView openBufferReadView();
}
