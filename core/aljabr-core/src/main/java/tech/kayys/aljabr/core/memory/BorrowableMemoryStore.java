package tech.kayys.aljabr.core.memory;

import java.util.Optional;

/**
 * Optional capability for stores that can safely expose provider-owned memory.
 *
 * <p>A successful borrow pins the underlying allocation until the returned
 * lease is closed. Providers that cannot offer this lifetime guarantee should
 * not implement this interface.
 */
public interface BorrowableMemoryStore extends UnifiedMemoryStore {

    /**
     * Borrows a value without copying it when the provider supports aliasing.
     *
     * @param key key to look up
     * @return a scoped lease, or empty when the key does not exist
     */
    Optional<MemoryStoreLease> borrow(byte[] key);
}
