package tech.kayys.aljabr.cache;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.quantization.TurboQuantProdKVCacheCodec;

import static org.junit.jupiter.api.Assertions.*;

class KVCacheAttentionBlockReaderTest {

    @Test
    void sessionDeliversAttentionVisibleRangeInStorageBlockSegments() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 9, 1, 4);
        BlockCompressedKVCacheStorage storage = new BlockCompressedKVCacheStorage(
                layout, new TurboQuantProdKVCacheCodec(4, 101L), 3);
        KVCacheSession session = new KVCacheSession(storage);
        float[] keys = vectors(6, 0.1f);
        float[] values = vectors(6, -0.3f);
        session.writeRange(0, 0, 6, keys, 0, values, 0);
        session.commit(6);

        List<Integer> positions = new ArrayList<>();
        List<Integer> counts = new ArrayList<>();
        List<float[]> receivedKeys = new ArrayList<>();
        List<float[]> receivedValues = new ArrayList<>();
        List<Integer> offsets = new ArrayList<>();
        session.readBlocksForAttention(0, 1, 4,
                (first, count, blockKeys, blockValues, offset) -> {
                    positions.add(first);
                    counts.add(count);
                    receivedKeys.add(blockKeys.clone());
                    receivedValues.add(blockValues.clone());
                    offsets.add(offset);
                    assertTrue(offset >= 0);
                    assertTrue(offset + count * 4 <= blockKeys.length);
                    assertTrue(offset + count * 4 <= blockValues.length);
                });

        assertEquals(List.of(1, 3), positions);
        assertEquals(List.of(2, 2), counts);
        assertEquals(List.of(4, 0), offsets);

        float[] expectedKeys = new float[16];
        float[] expectedValues = new float[16];
        storage.readKeyRange(0, 1, 4, expectedKeys, 0);
        storage.readValueRange(0, 1, 4, expectedValues, 0);
        int target = 0;
        for (int block = 0; block < positions.size(); block++) {
            int count = counts.get(block);
            int offset = offsets.get(block);
            for (int token = 0; token < count; token++) {
                for (int d = 0; d < 4; d++) {
                    assertEquals(expectedKeys[target + d],
                            receivedKeys.get(block)[offset + token * 4 + d], 1.0e-6f);
                    assertEquals(expectedValues[target + d],
                            receivedValues.get(block)[offset + token * 4 + d], 1.0e-6f);
                }
                target += 4;
            }
        }
        assertEquals(16, target);
    }

    @Test
    void attentionReadCannotExposeUnwrittenCurrentPosition() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 4);
        BlockCompressedKVCacheStorage storage = new BlockCompressedKVCacheStorage(
                layout, new TurboQuantProdKVCacheCodec(3, 103L), 4);
        KVCacheSession session = new KVCacheSession(storage);
        float[] keys = vectors(3, 0.2f);
        float[] values = vectors(3, -0.2f);
        session.writeRange(0, 0, 3, keys, 0, values, 0);
        session.commit(2);

        assertThrows(IllegalStateException.class,
                () -> session.readBlocksForAttention(0, 0, 3,
                        (first, count, k, v, offset) -> fail("must not expose uncommitted token")));
    }

    @Test
    void compatibilityStorageUsesSingleRangeCallback() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 4);
        KVCacheStorage storage = new TestArrayStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        float[] keys = vectors(2, 0.15f);
        float[] values = vectors(2, -0.15f);
        session.writeRange(0, 0, 2, keys, 0, values, 0);
        session.commit(2);

        int[] callbacks = {0};
        session.readBlocksForAttention(0, 0, 2, (first, count, k, v, offset) -> {
            callbacks[0]++;
            assertEquals(0, first);
            assertEquals(2, count);
            assertEquals(0, offset);
            assertArrayEquals(keys, k);
            assertArrayEquals(values, v);
        });
        assertEquals(1, callbacks[0]);
    }

    private static float[] vectors(int tokens, float start) {
        float[] values = new float[tokens * 4];
        for (int i = 0; i < values.length; i++) {
            values[i] = start + (i + 1) * 0.03125f;
        }
        return values;
    }

    private static final class TestArrayStorage implements KVCacheStorage {
        private final KVCacheLayout layout;
        private final float[][] keys;
        private final float[][] values;

        private TestArrayStorage(KVCacheLayout layout) {
            this.layout = layout;
            this.keys = new float[layout.maxSequenceLength()][layout.layerStride(0)];
            this.values = new float[layout.maxSequenceLength()][layout.layerStride(0)];
        }

        @Override public KVCacheLayout layout() { return layout; }

        @Override
        public void write(int layer, int position, float[] key, int keyOffset,
                float[] value, int valueOffset) {
            System.arraycopy(key, keyOffset, keys[position], 0, 4);
            System.arraycopy(value, valueOffset, values[position], 0, 4);
        }

        @Override
        public void readKey(int layer, int position, float[] destination, int offset) {
            System.arraycopy(keys[position], 0, destination, offset, 4);
        }

        @Override
        public void readValue(int layer, int position, float[] destination, int offset) {
            System.arraycopy(values[position], 0, destination, offset, 4);
        }

        @Override public void clearLayer(int layer) { clear(); }
        @Override public void clear() {
            for (int i = 0; i < keys.length; i++) {
                java.util.Arrays.fill(keys[i], 0);
                java.util.Arrays.fill(values[i], 0);
            }
        }
        @Override public void truncate(int firstPosition) {
            for (int i = firstPosition; i < keys.length; i++) {
                java.util.Arrays.fill(keys[i], 0);
                java.util.Arrays.fill(values[i], 0);
            }
        }
        @Override public void close() { clear(); }
    }
}
