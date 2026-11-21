package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.cache.HeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheSession;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class GgufDecodeAttentionExecutionTest {
    @Test
    void mapsQueryHeadsToGroupedKvHeads() {
        float[] query = new float[4];
        float[] output = new float[4];
        float[] scores = new float[2];
        // [position][kv head], one scalar per head.
        float[] keys = { 0f, 0f, 0f, 0f };
        float[] values = { 10f, 100f, 20f, 200f };

        GgufDecodeAttentionExecution.execute(
                query, output, scores, keys, values,
                2, 2, 4, 1, 2, 0, 1f);

        // Query heads 0-1 share KV head 0; query heads 2-3 share KV head 1.
        assertArrayEquals(new float[] { 15f, 15f, 150f, 150f }, output, 1.0e-6f);
    }

    @Test
    void restrictsAttentionToCallerSelectedSlidingWindow() {
        float[] query = { 0f, 0f };
        float[] output = new float[2];
        float[] scores = new float[1];
        float[] keys = { 0f, 0f, 0f, 0f };
        float[] values = { 10f, 100f, 20f, 200f };

        GgufDecodeAttentionExecution.execute(
                query, output, scores, keys, values,
                2, 2, 2, 1, 2, 1, 1f);

        assertArrayEquals(new float[] { 20f, 200f }, output, 1.0e-6f);
    }
    @Test
    void readsCommittedPrefixAndCurrentLayerWriteThroughSession() {
        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(1, 4, 1, 1)));
        float[] key = {0f};
        float[] value = {10f};
        session.write(0, 0, key, 0, value, 0);
        value[0] = 20f;
        session.write(0, 1, key, 0, value, 0);
        session.commit(2);
        value[0] = 30f;
        session.write(0, 2, key, 0, value, 0);

        float[] output = new float[1];
        float[] scores = new float[2];
        GgufDecodeAttentionExecution.execute(
                new float[] {0f}, output, scores, session,
                0, 1, 1, 1, 3, 1, 1f);

        // Sliding window starts at position 1: average of values 20 and 30.
        assertArrayEquals(new float[] {25f}, output, 1.0e-6f);
        assertEquals(2, session.committedLength(),
                "attention must not commit the current decode position");
    }

    @Test
    void sessionAttentionRejectsUnwrittenCurrentPosition() {
        KVCacheSession session = new KVCacheSession(
                new HeapKVCacheStorage(KVCacheLayout.uniform(1, 4, 1, 1)));
        session.commit(1);

        assertThrows(IllegalStateException.class,
                () -> GgufDecodeAttentionExecution.execute(
                        new float[] {0f}, new float[1], new float[1], session,
                        0, 1, 1, 1, 2, 0, 1f));
    }
}
