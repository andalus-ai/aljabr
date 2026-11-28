package tech.kayys.aljabr.backend.hat;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheReadView;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HatKVCacheAttentionConsumerTest {

    @Test
    void stagedDecodeMatchesStableSoftmaxReference() {
        HatKVCacheAttentionConsumer consumer = new HatKVCacheAttentionConsumer();
        try (KVCacheReadView view = new ArrayReadView(
                new float[] { 0.0f, (float) Math.log(3.0) },
                new float[] { 2.0f, 6.0f })) {
            float[] output = consumer.decode(view, 0, new float[] { 1.0f },
                    1, 1, true, 1.0f);
            assertEquals(5.0f, output[0], 1e-4f);
        }
    }

    @Test
    void causalDecodeExcludesPositionsAfterQueryPosition() {
        HatKVCacheAttentionConsumer consumer = new HatKVCacheAttentionConsumer();
        try (KVCacheReadView view = new ArrayReadView(
                new float[] { 0.0f, (float) Math.log(3.0) },
                new float[] { 2.0f, 6.0f })) {
            float[] output = consumer.decode(view, 0, new float[] { 1.0f },
                    1, 0, true, 1.0f);
            assertEquals(2.0f, output[0], 1e-4f);
        }
    }

    private static final class ArrayReadView implements KVCacheReadView {
        private final KVCacheLayout layout = KVCacheLayout.uniform(1, 2, 1, 1);
        private final float[] keys;
        private final float[] values;
        private boolean closed;

        private ArrayReadView(float[] keys, float[] values) {
            this.keys = keys;
            this.values = values;
        }

        @Override
        public KVCacheLayout layout() {
            ensureOpen();
            return layout;
        }

        @Override
        public int visibleSequenceLength() {
            ensureOpen();
            return 2;
        }

        @Override
        public void readKeyRange(int layer, int firstPosition, int tokenCount,
                float[] destination, int destinationOffset) {
            ensureOpen();
            System.arraycopy(keys, firstPosition, destination, destinationOffset, tokenCount);
        }

        @Override
        public void readValueRange(int layer, int firstPosition, int tokenCount,
                float[] destination, int destinationOffset) {
            ensureOpen();
            System.arraycopy(values, firstPosition, destination, destinationOffset, tokenCount);
        }

        @Override
        public void close() {
            closed = true;
        }

        private void ensureOpen() {
            if (closed) throw new IllegalStateException("view is closed");
        }
    }
}
