package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

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
}
