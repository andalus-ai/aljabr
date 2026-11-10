package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufKvCacheWriteExecutionTest {

    @Test
    void tokenWriteUsesCurrentSequencePositionAndDoesNotAdvanceIt() {
        KVCache cache = new KVCache(1, 4, 1, 2);
        float[] key0 = {1f, 2f};
        float[] value0 = {10f, 20f};
        GgufKvCacheWriteExecution.storeToken(cache, 0, key0, value0);

        assertEquals(0, cache.seqLen(), "writing one layer must not advance the sequence");
        assertArrayEquals(key0, slice(cache.kLayer(0), 0, 2));
        assertArrayEquals(value0, slice(cache.vLayer(0), 0, 2));

        cache.advance();
        float[] key1 = {3f, 4f};
        float[] value1 = {30f, 40f};
        GgufKvCacheWriteExecution.storeToken(cache, 0, key1, value1);

        assertEquals(1, cache.seqLen(), "the caller advances after all layer writes");
        assertArrayEquals(key0, slice(cache.kLayer(0), 0, 2));
        assertArrayEquals(key1, slice(cache.kLayer(0), 2, 2));
        assertArrayEquals(value0, slice(cache.vLayer(0), 0, 2));
        assertArrayEquals(value1, slice(cache.vLayer(0), 2, 2));
    }

    @Test
    void batchWriteHonorsExplicitPositionAndOffsetsWithoutAdvancing() {
        KVCache cache = new KVCache(1, 5, 1, 2);
        float[] keys = {-1f, -1f, 1f, 2f, 3f, 4f, -1f, -1f};
        float[] values = {-2f, -2f, 10f, 20f, 30f, 40f, -2f, -2f};

        GgufKvCacheWriteExecution.storeBatch(cache, 0, 1, keys, 2, values, 2, 4);

        assertEquals(0, cache.seqLen(), "batch writes do not own sequence advancement");
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, slice(cache.kLayer(0), 2, 4));
        assertArrayEquals(new float[] {10f, 20f, 30f, 40f}, slice(cache.vLayer(0), 2, 4));

        cache.advance(3);
        assertEquals(3, cache.seqLen());
    }

    private static float[] slice(float[] source, int offset, int length) {
        float[] result = new float[length];
        System.arraycopy(source, offset, result, 0, length);
        return result;
    }
}
