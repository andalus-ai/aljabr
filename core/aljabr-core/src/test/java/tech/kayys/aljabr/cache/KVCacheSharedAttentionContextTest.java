package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KVCacheSharedAttentionContextTest {

    @Test
    void borrowedViewReadsCommittedPrefixWithoutOwningStorage() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 5, 1, 2));
        KVCacheSession session = new KVCacheSession(storage);
        float[] key = {1, 2};
        float[] value = {3, 4};
        session.write(0, 0, key, 0, value, 0);
        session.write(0, 1, key, 0, value, 0);
        session.commit(2);

        KVCacheReadView view = session.openReadView();
        float[] actualKey = new float[4];
        float[] actualValue = new float[4];
        view.readKeyRange(0, 0, 2, actualKey, 0);
        view.readValueRange(0, 0, 2, actualValue, 0);

        assertArrayEquals(new float[] {1, 2, 1, 2}, actualKey);
        assertArrayEquals(new float[] {3, 4, 3, 4}, actualValue);
        assertEquals(2, view.visibleSequenceLength());
        assertThrows(IndexOutOfBoundsException.class,
                () -> view.readKeyRange(0, 0, 3, new float[6], 0));

        view.close();
        assertDoesNotThrow(storage::clear);
        storage.close();
    }

    @Test
    void viewIsBoundedToOpeningSnapshotAndAllowsAppending() {
        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(1, 5, 1, 1)));
        float[] pair = {7};
        session.write(0, 0, pair, 0, pair, 0);
        session.commit(1);

        KVCacheReadView view = session.openReadView();
        session.write(0, 1, pair, 0, pair, 0);
        session.commit(2);

        assertEquals(1, view.visibleSequenceLength());
        assertThrows(IndexOutOfBoundsException.class,
                () -> view.readKeyRange(0, 0, 2, new float[2], 0));
        float[] visible = new float[1];
        view.readKeyRange(0, 0, 1, visible, 0);
        assertArrayEquals(pair, visible);
        view.close();
    }

    @Test
    void protectsBorrowedPrefixAndRejectsStructuralMutationUntilClosed() {
        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(1, 4, 1, 1)));
        float[] pair = {5};
        session.write(0, 0, pair, 0, pair, 0);
        session.commit(1);
        KVCacheReadView view = session.openReadView();

        assertThrows(IllegalStateException.class,
                () -> session.write(0, 0, pair, 0, pair, 0));
        assertThrows(IllegalStateException.class, () -> session.truncate(0));
        assertThrows(IllegalStateException.class, session::reset);

        view.close();
        assertDoesNotThrow(() -> session.write(0, 0, pair, 0, pair, 0));
        assertDoesNotThrow(() -> session.truncate(0));
    }

    @Test
    void closingViewIsIdempotentAndDisablesFurtherReads() {
        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(1, 2, 1, 1)));
        session.commit(1);
        KVCacheReadView view = session.openReadView();

        view.close();
        view.close();
        assertThrows(IllegalStateException.class,
                () -> view.readKeyRange(0, 0, 1, new float[1], 0));
        assertThrows(IllegalStateException.class, view::visibleSequenceLength);
    }

    @Test
    void selectionIsImmutablePreservesOrderAndValidatesCachePrefix() {
        int[] source = {3, 0, 2};
        AttentionIndexSelection selection = new AttentionIndexSelection(4, source);
        source[0] = 1;

        assertEquals(3, selection.size());
        assertEquals(3, selection.positionAt(0));
        assertArrayEquals(new int[] {3, 0, 2}, selection.positions());

        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(1, 5, 1, 1)));
        session.commit(3);
        try (KVCacheReadView view = session.openReadView()) {
            assertThrows(IllegalArgumentException.class,
                    () -> selection.validateAgainst(view));
        }
    }

    @Test
    void selectionRejectsDuplicateAndOutOfRangePositions() {
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionIndexSelection(3, new int[] {0, 1, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionIndexSelection(3, new int[] {3}));
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionIndexSelection(-1, new int[0]));
    }
}
