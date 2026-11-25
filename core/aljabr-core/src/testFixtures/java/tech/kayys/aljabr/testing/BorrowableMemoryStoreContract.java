package tech.kayys.aljabr.testing;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Optional;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.BorrowableMemoryStore;
import tech.kayys.aljabr.core.memory.MemoryStoreLease;

/**
 * Reusable contract for stores that expose provider-owned memory through
 * scoped {@link MemoryStoreLease} instances.
 *
 * <p>Implementations supply a fresh store for each test. The contract checks
 * lease semantics only; provider-specific synchronization and reclamation
 * guarantees should be covered by additional tests.</p>
 */
public abstract class BorrowableMemoryStoreContract {

    protected abstract BorrowableMemoryStore newStore();

    @Test
    public final void missingKeyReturnsEmptyBorrow() {
        try (BorrowableMemoryStore store = newStore()) {
            assertTrue(store.borrow("missing".getBytes()).isEmpty());
        }
    }

    @Test
    public final void borrowedValueIsReadableAndLeaseCloseEndsLeaseAccess() {
        byte[] key = "borrow-contract-value".getBytes();
        long expected = 0x1020304050607080L;

        try (BorrowableMemoryStore store = newStore();
                Arena sourceArena = Arena.ofShared()) {
            MemorySegment source = sourceArena.allocate(Long.BYTES);
            source.set(ValueLayout.JAVA_LONG, 0, expected);
            store.put(key, source);

            Optional<MemoryStoreLease> result = store.borrow(key);
            assertTrue(result.isPresent());

            MemoryStoreLease lease = result.orElseThrow();
            assertFalse(lease.isClosed());
            assertEquals(Long.BYTES, lease.segment().byteSize());
            assertEquals(expected, lease.segment().get(ValueLayout.JAVA_LONG, 0));

            lease.close();

            assertTrue(lease.isClosed());
            assertThrows(IllegalStateException.class, lease::segment);
        }
    }

    @Test
    public final void independentLeasesHaveIndependentCloseLifetimes() {
        byte[] key = "independent-leases".getBytes();
        byte[] expected = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };

        try (BorrowableMemoryStore store = newStore();
                Arena sourceArena = Arena.ofShared()) {
            MemorySegment source = sourceArena.allocate(expected.length);
            for (int i = 0; i < expected.length; i++) {
                source.set(ValueLayout.JAVA_BYTE, i, expected[i]);
            }
            store.put(key, source);

            MemoryStoreLease first = store.borrow(key).orElseThrow();
            MemoryStoreLease second = store.borrow(key).orElseThrow();

            first.close();
            first.close();

            assertTrue(first.isClosed());
            assertFalse(second.isClosed());
            byte[] actual = new byte[expected.length];
            MemorySegment.copy(second.segment(), 0, MemorySegment.ofArray(actual), 0, actual.length);
            assertArrayEquals(expected, actual);

            second.close();
            assertTrue(second.isClosed());
            assertThrows(IllegalStateException.class, second::segment);
        }
    }
}
