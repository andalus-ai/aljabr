package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.inference.KVCache;

/**
 * Backend-neutral host KV-cache write operations used by GGUF execution.
 *
 * <p>These methods only write the supplied K/V values. They deliberately do
 * not advance the cache sequence pointer: callers advance once after all
 * layers have been populated for the corresponding token range.</p>
 */
final class GgufKvCacheWriteExecution {
    private GgufKvCacheWriteExecution() {
    }

    static void storeToken(KVCache cache, int layer, float[] key, float[] value) {
        cache.store(layer, key, value);
    }

    static void storeBatch(
            KVCache cache,
            int layer,
            int startPosition,
            float[] keys,
            int keysOffset,
            float[] values,
            int valuesOffset,
            int count) {
        cache.storeBatch(layer, startPosition, keys, keysOffset, values, valuesOffset, count);
    }
}
