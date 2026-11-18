package tech.kayys.aljabr.gguf.loader.inference;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.cache.KVCacheStorage;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufKVCacheStorageAdapterTest {

    @Test
    void bulkStorageAccessUsesExistingGgufBackingArraysAndHybridGeometry() {
        KVCache cache = new KVCache(2, 4, new int[] {1, 2}, new int[] {2, 2});
        KVCacheStorage storage = new GgufKVCacheStorageAdapter(cache);
        float[] keys = {90, 1, 2, 3, 4, 5, 6, 91};
        float[] values = {80, 11, 12, 13, 14, 15, 16, 81};

        storage.writeRange(0, 1, 2, keys, 1, values, 1);
        storage.writeRange(1, 0, 2, keys, 1, values, 1);

        assertArrayEquals(new float[] {0, 1, 2, 3, 4, 5, 6, 0}, cache.kLayer(0));
        assertArrayEquals(new float[] {11, 12, 13, 14, 15, 16, 0, 0}, cache.vLayer(0));
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 0, 0}, cache.kLayer(1));
        assertEquals(0, cache.seqLen(), "storage writes must not commit sequence positions");

        float[] readKeys = new float[8];
        float[] readValues = new float[8];
        storage.readKeyRange(0, 1, 2, readKeys, 2);
        storage.readValueRange(0, 1, 2, readValues, 2);
        assertArrayEquals(new float[] {0, 0, 1, 2, 3, 4, 0, 0}, readKeys);
        assertArrayEquals(new float[] {0, 0, 11, 12, 13, 14, 0, 0}, readValues);
    }

    @Test
    void clearAndTruncateOnlyChangeStorageNotSequenceLifecycle() {
        KVCache cache = new KVCache(1, 3, 1, 2);
        KVCacheStorage storage = new GgufKVCacheStorageAdapter(cache);
        storage.writeRange(0, 0, 3,
                new float[] {1, 2, 3, 4, 5, 6}, 0,
                new float[] {7, 8, 9, 10, 11, 12}, 0);
        cache.advance(2);

        storage.truncate(1);
        assertArrayEquals(new float[] {1, 2, 0, 0, 0, 0}, cache.kLayer(0));
        assertArrayEquals(new float[] {7, 8, 0, 0, 0, 0}, cache.vLayer(0));
        assertEquals(2, cache.seqLen());

        storage.clear();
        assertArrayEquals(new float[6], cache.kLayer(0));
        assertArrayEquals(new float[6], cache.vLayer(0));
        assertEquals(2, cache.seqLen());
    }

    @Test
    void invalidBulkWriteDoesNotMutateEitherBackingArray() {
        KVCache cache = new KVCache(1, 2, 1, 2);
        KVCacheStorage storage = new GgufKVCacheStorageAdapter(cache);
        cache.storeDirect(0, 0, new float[] {21, 22}, new float[] {31, 32}, 2);

        assertThrows(IndexOutOfBoundsException.class,
                () -> storage.writeRange(0, 0, 1,
                        new float[] {1, 2}, 0, new float[] {3}, 0));

        assertArrayEquals(new float[] {21, 22, 0, 0}, cache.kLayer(0));
        assertArrayEquals(new float[] {31, 32, 0, 0}, cache.vLayer(0));
    }

    @Test
    void closeOnlyClosesAdapterAndLeavesWrappedCacheUsable() {
        KVCache cache = new KVCache(1, 2, 1, 2);
        GgufKVCacheStorageAdapter storage = new GgufKVCacheStorageAdapter(cache);
        storage.close();

        assertThrows(IllegalStateException.class,
                () -> storage.readKey(0, 0, new float[2], 0));
        cache.storeDirect(0, 0, new float[] {4, 5}, new float[] {6, 7}, 2);
        assertArrayEquals(new float[] {4, 5, 0, 0}, cache.kLayer(0));
    }

    @Test
    void rejectsOutOfCapacityRanges() {
        KVCacheStorage storage = new GgufKVCacheStorageAdapter(new KVCache(1, 2, 1, 2));
        assertThrows(IndexOutOfBoundsException.class,
                () -> storage.readKeyRange(0, 2, 1, new float[2], 0));
        assertThrows(IndexOutOfBoundsException.class,
                () -> storage.writeRange(0, -1, 1,
                        new float[2], 0, new float[2], 0));
    }
}
