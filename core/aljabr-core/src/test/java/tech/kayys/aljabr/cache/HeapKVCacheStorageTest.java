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
    void bulkRangeAccessPreservesOffsetsAndHybridLayerStride() {
        // Layer 0: 1 head × 2 dim = stride 2
        // Layer 1: 2 heads × 2 dim = stride 4
        // writeRange on layer 1 with tokenCount=2 needs 2×4 = 8 floats
        KVCacheLayout layout = new KVCacheLayout(
                5, new int[] {1, 2}, new int[] {2, 2});
        try (KVCacheStorage storage = new HeapKVCacheStorage(layout)) {
            float[] keys = {-1, 1, 2, 3, 4, 5, 6, 7, 8, -1};
            float[] values = {-1, 11, 12, 13, 14, 15, 16, 17, 18, -1};
            storage.writeRange(1, 2, 2, keys, 1, values, 1);

            float[] actualKeys = new float[12];
            float[] actualValues = new float[12];
            storage.readKeyRange(1, 2, 2, actualKeys, 3);
            storage.readValueRange(1, 2, 2, actualValues, 3);

            assertArrayEquals(new float[] {0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0}, actualKeys);
            assertArrayEquals(new float[] {0, 0, 0, 11, 12, 13, 14, 15, 16, 17, 18, 0}, actualValues);
        }
    }

    @Test
    void bulkWriteValidatesBothSourcesBeforeMutatingStorage() {
        KVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 3, 1, 2));
        float[] initial = {7, 8, 9, 10};
        storage.writeRange(0, 0, 2, initial, 0, initial, 0);

        assertThrows(IndexOutOfBoundsException.class,
                () -> storage.writeRange(0, 0, 2,
                        new float[] {1, 2, 3, 4}, 0,
                        new float[] {5, 6, 7}, 0));

        float[] actualKeys = new float[4];
        float[] actualValues = new float[4];
        storage.readKeyRange(0, 0, 2, actualKeys, 0);
        storage.readValueRange(0, 0, 2, actualValues, 0);
        assertArrayEquals(initial, actualKeys);
        assertArrayEquals(initial, actualValues);
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
