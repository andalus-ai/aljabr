package tech.kayys.aljabr.cache;

/**
 * Internal storage-side hook for opening backend-native KV buffer leases.
 *
 * <p>The visible length is supplied by the session, which owns logical
 * commit/visibility state. Implementations must enforce the lease against
 * direct storage mutations as well as session-mediated writes.</p>
 */
interface KVCacheBufferRegionProvider {
    KVCacheBufferReadView openBufferReadView(int visibleSequenceLength);
}
