package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class GgufKvCacheReadExecutionTest {

    @Test
    void copiesContiguousSharedLayerWhenStridesMatch() {
        KVCache cache = new KVCache(1, 3, 2, 2);
        float[] keys = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] values = {10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f};
        cache.storeBatch(0, 0, keys, 0, values, 0, keys.length);

        float[] targetKeys = new float[8];
        float[] targetValues = new float[8];
        GgufKvCacheReadExecution.copyLayerToBatch(cache, 0, targetKeys, targetValues, 2, 4);

        assertArrayEquals(keys, targetKeys);
        assertArrayEquals(values, targetValues);
    }

    @Test
    void compactsEachTokenWhenSharedLayerStrideIsWider() {
        KVCache cache = new KVCache(1, 3, 3, 1);
        float[] keys = {1f, 2f, 99f, 3f, 4f, 99f};
        float[] values = {10f, 20f, 99f, 30f, 40f, 99f};
        cache.storeBatch(0, 0, keys, 0, values, 0, keys.length);

        float[] targetKeys = new float[] {-1f, -1f, -1f, -1f};
        float[] targetValues = new float[] {-2f, -2f, -2f, -2f};
        GgufKvCacheReadExecution.copyLayerToBatch(cache, 0, targetKeys, targetValues, 2, 2);

        assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, targetKeys);
        assertArrayEquals(new float[] {10f, 20f, 30f, 40f}, targetValues);
    }
}
