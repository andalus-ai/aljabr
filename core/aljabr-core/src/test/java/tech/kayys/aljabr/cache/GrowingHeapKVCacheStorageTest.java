package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GrowingHeapKVCacheStorageTest {

    @Test
    void defaultInitialCapacityIsCappedByShortLogicalContext() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 4, 1, 2);
        try (GrowingHeapKVCacheStorage storage = new GrowingHeapKVCacheStorage(layout)) {
            assertEquals(4, storage.allocatedPositions());
        }
    }

    @Test
    void allocatesOnlyInitialCapacityAndGrowsForWrites() {
        KVCacheLayout layout = KVCacheLayout.uniform(2, 64, 2, 4);
        try (GrowingHeapKVCacheStorage storage = new GrowingHeapKVCacheStorage(layout, 2)) {
            assertEquals(2, storage.allocatedPositions());
            float[] key = {1, 2, 3, 4, 5, 6, 7, 8};
            float[] value = {11, 12, 13, 14, 15, 16, 17, 18};

            storage.write(1, 5, key, 0, value, 0);

            assertEquals(6, storage.allocatedPositions());
            float[] actualKey = new float[8];
            float[] actualValue = new float[8];
            storage.readKey(1, 5, actualKey, 0);
            storage.readValue(1, 5, actualValue, 0);
            assertArrayEquals(key, actualKey);
            assertArrayEquals(value, actualValue);
        }
    }

    @Test
    void growsWithHybridLayerGeometryAndSupportsBulkRanges() {
        KVCacheLayout layout = new KVCacheLayout(
                32, new int[] {1, 2}, new int[] {2, 2});
        try (GrowingHeapKVCacheStorage storage = new GrowingHeapKVCacheStorage(layout, 0)) {
            float[] keys = {1, 2, 3, 4, 5, 6, 7, 8};
            float[] values = {11, 12, 13, 14, 15, 16, 17, 18};

            storage.writeRange(1, 3, 2, keys, 0, values, 0);

            assertEquals(5, storage.allocatedPositions());
            float[] actualKeys = new float[8];
            float[] actualValues = new float[8];
            storage.readKeyRange(1, 3, 2, actualKeys, 0);
            storage.readValueRange(1, 3, 2, actualValues, 0);
            assertArrayEquals(keys, actualKeys);
            assertArrayEquals(values, actualValues);
        }
    }

    @Test
    void rejectsReadsOutsideAllocatedCapacityAndWritesOutsideLogicalCapacity() {
        try (GrowingHeapKVCacheStorage storage = new GrowingHeapKVCacheStorage(
                KVCacheLayout.uniform(1, 8, 1, 2), 1)) {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> storage.readKey(0, 1, new float[2], 0));
            assertThrows(IndexOutOfBoundsException.class,
                    () -> storage.write(0, 8, new float[2], 0, new float[2], 0));
        }
    }

    @Test
    void clearAndTruncateZeroValuesWithoutShrinkingAllocatedCapacity() {
        try (GrowingHeapKVCacheStorage storage = new GrowingHeapKVCacheStorage(
                KVCacheLayout.uniform(2, 8, 1, 2), 1)) {
            float[] pair = {7, 8};
            for (int layer = 0; layer < 2; layer++) {
                storage.write(layer, 3, pair, 0, pair, 0);
            }
            assertEquals(4, storage.allocatedPositions());

            storage.truncate(3);
            float[] cleared = new float[2];
            storage.readKey(1, 3, cleared, 0);
            assertArrayEquals(new float[] {0, 0}, cleared);

            storage.write(0, 3, pair, 0, pair, 0);
            storage.clearLayer(0);
            storage.readValue(0, 3, cleared, 0);
            assertArrayEquals(new float[] {0, 0}, cleared);
            float[] retained = new float[2];
            storage.readValue(1, 3, retained, 0);
            assertArrayEquals(new float[] {0, 0}, retained);
            assertEquals(4, storage.allocatedPositions());
        }
    }


    @Test
    void worksWithCommittedSequenceSession() {
        GrowingHeapKVCacheStorage storage = new GrowingHeapKVCacheStorage(
                KVCacheLayout.uniform(1, 32, 1, 2), 0);
        KVCacheSession session = new KVCacheSession(storage);
        float[] key = {1, 2};
        float[] value = {3, 4};

        session.write(0, 0, key, 0, value, 0);
        session.write(0, 1, key, 0, value, 0);
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readKey(0, 0, new float[2], 0));

        session.commit(2);
        float[] actual = new float[2];
        session.readValue(0, 1, actual, 0);
        assertArrayEquals(value, actual);

        session.truncate(1);
        assertEquals(1, session.committedLength());
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readKey(0, 1, new float[2], 0));
        storage.close();
    }

    @Test
    void closeReleasesBackingArraysAndRejectsFurtherAccess() {
        GrowingHeapKVCacheStorage storage = new GrowingHeapKVCacheStorage(
                KVCacheLayout.uniform(1, 8, 1, 2), 1);
        storage.close();

        assertThrows(IllegalStateException.class, storage::allocatedPositions);
        assertThrows(IllegalStateException.class,
                () -> storage.write(0, 0, new float[2], 0, new float[2], 0));
        assertThrows(IllegalStateException.class, storage::clear);
    }
}
