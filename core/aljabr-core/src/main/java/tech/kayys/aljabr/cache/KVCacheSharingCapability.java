package tech.kayys.aljabr.cache;

/**
 * Optional capability for exposing a target-owned KV cache to read-only
 * consumers such as speculative drafters.
 *
 * <p>Implementations must define view lifetime, synchronization, visibility,
 * and mutation rules. The returned view must not transfer storage ownership
 * to the consumer. Implementations that cannot safely provide a borrowed
 * view must not advertise this capability.</p>
 */
public interface KVCacheSharingCapability {

    /**
     * Opens a borrowed view bounded to the currently committed sequence prefix.
     *
     * @return a caller-closeable read-only view
     */
    KVCacheReadView openReadView();
}
