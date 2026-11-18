package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HeapKVCacheStorageTest {

    @Test
    void writesAndReadsIndependentKeyAndValuePositions() {
        KVCacheLayout layout = KVCacheLayout.uniform(2, 4, 2, 2);
        try (KVCacheStorage storage = new HeapKVCacheStorage(layout)) {
            float[] key = {1, 2, 3, 4, 90};
            float[] value = {5, 6, 7, 8, 90};
            storage.write(1, 2, key, 0, value, 0);

            float[] actualKey = new float[4];
            float[] actualValue = new float[4];
            storage.readKey(1, 2, actualKey, 0);
            storage.readValue(1, 2, actualValue, 0);

            assertArrayEquals(new float[] {1, 2, 3, 4}, actualKey);
            assertArrayEquals(new float[] {5, 6, 7, 8}, actualValue);
        }
    }

    @Test
    void validatesBothInputRangesBeforeMutatingEitherCache() {
        KVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 2, 1, 2));
        float[] initial = {11, 12};
        storage.write(0, 0, initial, 0, initial, 0);

        assertThrows(IndexOutOfBoundsException.class,
                () -> storage.write(0, 0, new float[] {1, 2}, 0,
                        new float[] {3}, 0));

        float[] key = new float[2];
        float[] value = new float[2];
        storage.readKey(0, 0, key, 0);
        storage.readValue(0, 0, value, 0);
        assertArrayEquals(initial, key);
        assertArrayEquals(initial, value);
    }

    @Test
    void truncateClearsTailAcrossAllLayers() {
        KVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(2, 4, 1, 2));
        float[] pair = {1, 2};
        for (int layer = 0; layer < 2; layer++) {
            for (int position = 0; position < 4; position++) {
                storage.write(layer, position, pair, 0, pair, 0);
            }
        }

        storage.truncate(2);
        float[] retained = new float[2];
        float[] cleared = new float[2];
        storage.readKey(1, 1, retained, 0);
        storage.readValue(1, 3, cleared, 0);

        assertArrayEquals(pair, retained);
        assertArrayEquals(new float[] {0, 0}, cleared);
    }

    @Test
    void clearLayerDoesNotClearOtherLayers() {
        KVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(2, 2, 1, 2));
        float[] pair = {7, 8};
        storage.write(0, 0, pair, 0, pair, 0);
        storage.write(1, 0, pair, 0, pair, 0);

        storage.clearLayer(0);
        float[] cleared = new float[2];
        float[] retained = new float[2];
        storage.readKey(0, 0, cleared, 0);
        storage.readKey(1, 0, retained, 0);

        assertArrayEquals(new float[] {0, 0}, cleared);
        assertArrayEquals(pair, retained);
    }

    @Test
    void rejectsInvalidLayerPositionAndUseAfterClose() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 2, 1, 2));

        assertThrows(IndexOutOfBoundsException.class,
                () -> storage.readKey(1, 0, new float[2], 0));
        assertThrows(IndexOutOfBoundsException.class,
                () -> storage.readKey(0, 2, new float[2], 0));

        storage.close();
        assertThrows(IllegalStateException.class,
                () -> storage.readKey(0, 0, new float[2], 0));
    }
}
