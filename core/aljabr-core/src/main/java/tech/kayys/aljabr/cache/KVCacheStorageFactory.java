package tech.kayys.aljabr.cache;

/**
 * Creates storage for a model sequence's key/value cache.
 *
 * <p>Factories receive the complete logical geometry and return a storage
 * implementation compatible with that layout. A factory may select heap,
 * direct/native, or device-backed storage according to its execution context.
 * The caller owns the returned storage and must close it.</p>
 */
@FunctionalInterface
public interface KVCacheStorageFactory {

    /** Creates cache storage for the supplied logical layout. */
    KVCacheStorage create(KVCacheLayout layout);
}
