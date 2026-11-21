package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KVCacheSessionTest {

    @Test
    void writesUncommittedPositionsButExposesOnlyCommittedPrefix() {
        KVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(2, 5, 1, 2));
        KVCacheSession session = new KVCacheSession(storage);
        float[] key = {1, 2};
        float[] value = {3, 4};

        session.write(0, 0, key, 0, value, 0);
        session.write(1, 0, key, 0, value, 0);
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readKey(0, 0, new float[2], 0));

        session.commit(1);
        float[] actual = new float[2];
        session.readKey(0, 0, actual, 0);
        assertArrayEquals(key, actual);
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readValue(0, 1, new float[2], 0));
    }

    @Test
    void attentionCanReadCurrentLayerWriteBeforeSequenceCommit() {
        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(2, 4, 1, 2)));
        float[] key = {1, 2};
        float[] value = {3, 4};
        session.write(0, 0, key, 0, value, 0);

        float[] actualKey = new float[2];
        float[] actualValue = new float[2];
        session.readKeyRangeForAttention(0, 0, 1, actualKey, 0);
        session.readValueRangeForAttention(0, 0, 1, actualValue, 0);
        assertArrayEquals(key, actualKey);
        assertArrayEquals(value, actualValue);
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readKey(0, 0, new float[2], 0));
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readKeyRangeForAttention(0, 0, 2, new float[4], 0));
    }

    @Test
    void supportsHybridLayerGeometryAndBulkPrefill() {
        KVCacheLayout layout = new KVCacheLayout(
                4, new int[] {1, 2}, new int[] {2, 2});
        KVCacheSession session = new KVCacheSession(new HeapKVCacheStorage(layout));
        float[] keys = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] values = {11, 12, 13, 14, 15, 16, 17, 18};

        session.writeRange(1, 0, 2, keys, 0, values, 0);
        session.commit(2);

        float[] actual = new float[8];
        session.readKeyRange(1, 0, 2, actual, 0);
        assertArrayEquals(keys, actual);
    }

    @Test
    void truncateClearsDiscardedSuffixAndMovesCommitBoundary() {
        KVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 4, 1, 2));
        KVCacheSession session = new KVCacheSession(storage);
        float[] pair = {7, 8};
        for (int position = 0; position < 3; position++) {
            session.write(0, position, pair, 0, pair, 0);
        }
        session.commit(3);

        session.truncate(1);
        assertEquals(1, session.committedLength());
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readKey(0, 1, new float[2], 0));
        float[] discarded = new float[2];
        storage.readKey(0, 2, discarded, 0);
        assertArrayEquals(new float[] {0, 0}, discarded);
    }

    @Test
    void resetClearsStorageAndDoesNotCloseCallerOwnedStorage() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 2, 1, 2));
        KVCacheSession session = new KVCacheSession(storage);
        float[] pair = {5, 6};
        session.write(0, 0, pair, 0, pair, 0);
        session.commit(1);

        session.reset();
        assertEquals(0, session.committedLength());
        storage.readKey(0, 0, new float[2], 0);
        assertDoesNotThrow(storage::clear);
        storage.close();
    }

    @Test
    void rejectsInvalidCommitAndTruncationWithoutChangingState() {
        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(1, 3, 1, 2)));
        session.commit(2);

        assertThrows(IllegalArgumentException.class, () -> session.commit(1));
        assertThrows(IllegalArgumentException.class, () -> session.commit(4));
        assertThrows(IllegalArgumentException.class, () -> session.truncate(3));
        assertEquals(2, session.committedLength());
    }
}
