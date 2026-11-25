package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.cache.BlockCompressedKVCacheStorage;
import tech.kayys.aljabr.cache.HeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.cache.quantization.TurboQuantProdKVCacheCodec;

import static org.junit.jupiter.api.Assertions.*;

class CpuKVCacheStreamingAttentionTest {

    @Test
    void computesCausalGqaAttentionAcrossCompressedBlocks() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 2, 2);
        BlockCompressedKVCacheStorage storage = new BlockCompressedKVCacheStorage(
                layout, new TurboQuantProdKVCacheCodec(4, 211L), 2);
        KVCacheSession session = new KVCacheSession(storage);

        float[] keys = {
                0.2f, 0.1f, 1.0f, -0.2f,
                0.4f, 0.3f, 0.7f, 0.8f,
                -0.1f, 0.5f, -0.3f, 1.1f
        };
        float[] values = {
                1.0f, 2.0f, 10.0f, 20.0f,
                3.0f, 4.0f, 30.0f, 40.0f,
                5.0f, 6.0f, 50.0f, 60.0f
        };
        session.writeRange(0, 0, 3, keys, 0, values, 0);
        session.commit(3);

        float[] query = {0.6f, -0.4f};
        float[] actual = CpuKVCacheStreamingAttention.compute(
                session, 0, query, 2, 4, 1, 0, 3, true);

        float[] decodedKeys = new float[12];
        float[] decodedValues = new float[12];
        storage.readKeyRange(0, 0, 3, decodedKeys, 0);
        storage.readValueRange(0, 0, 3, decodedValues, 0);
        float[] expected = reference(query, decodedKeys, decodedValues, 2, 1, 2);

        assertArrayEquals(expected, actual, 1.0e-5f);
        assertTrue(actual[0] > 10.0f && actual[0] < 30.0f,
                "query head 2 must map to KV head 1, not KV head 0");
    }

    @Test
    void rejectsUncommittedCacheSuffix() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 4, 1, 2);
        KVCacheSession session = new KVCacheSession(new HeapKVCacheStorage(layout));
        session.write(0, 0, new float[] {1, 0}, 0, new float[] {2, 3}, 0);

        assertThrows(IndexOutOfBoundsException.class,
                () -> CpuKVCacheStreamingAttention.compute(
                        session, 0, new float[] {1, 0}, 0, 1, 0, 0, 2, true));
    }

    private static float[] reference(
            float[] query,
            float[] keys,
            float[] values,
            int keyCount,
            int kvHead,
            int headDimension) {
        double[] scores = new double[keyCount];
        double max = Double.NEGATIVE_INFINITY;
        int stride = 2 * headDimension;
        int headOffset = kvHead * headDimension;
        for (int token = 0; token < keyCount; token++) {
            double score = 0;
            for (int d = 0; d < headDimension; d++) {
                score += query[d] * keys[token * stride + headOffset + d];
            }
            scores[token] = score / Math.sqrt(headDimension);
            max = Math.max(max, scores[token]);
        }
        double denominator = 0;
        double[] output = new double[headDimension];
        for (int token = 0; token < keyCount; token++) {
            double weight = Math.exp(scores[token] - max);
            denominator += weight;
            for (int d = 0; d < headDimension; d++) {
                output[d] += weight * values[token * stride + headOffset + d];
            }
        }
        float[] result = new float[headDimension];
        for (int d = 0; d < headDimension; d++) {
            result[d] = (float) (output[d] / denominator);
        }
        return result;
    }
}
