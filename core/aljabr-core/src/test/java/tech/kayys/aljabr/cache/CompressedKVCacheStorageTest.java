package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.quantization.TurboQuantProdKVCacheCodec;

import static org.junit.jupiter.api.Assertions.*;

class CompressedKVCacheStorageTest {

    @Test
    void roundTripsLayerSpecificKeyValueGeometryAndRangeOperations() {
        KVCacheLayout layout = new KVCacheLayout(
                8, new int[]{1, 2}, new int[]{4, 5});
        CompressedKVCacheStorage storage =
                new CompressedKVCacheStorage(layout, new TurboQuantProdKVCacheCodec(4, 71L));

        float[] layerZeroKeys = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] layerZeroValues = {-0.1f, 0.3f, 0.7f, -0.4f};
        float[] layerOneKeys = {0.2f, -0.1f, 0.4f, 0.8f, 0.5f, -0.7f, 0.1f, 0.3f, -0.2f, 0.6f};
        float[] layerOneValues = {-0.4f, 0.3f, 0.2f, 0.1f, 0.7f, 0.5f, -0.2f, 0.4f, 0.8f, -0.3f};

        storage.write(0, 0, layerZeroKeys, 0, layerZeroValues, 0);
        storage.write(1, 2, layerOneKeys, 0, layerOneValues, 0);

        float[] decodedKeys = new float[10];
        float[] decodedValues = new float[10];
        storage.readKey(1, 2, decodedKeys, 0);
        storage.readValue(1, 2, decodedValues, 0);

        assertFiniteAndNonZero(decodedKeys);
        assertFiniteAndNonZero(decodedValues);
        assertEquals(2, storage.storedPositionCount());
        assertTrue(storage.encodedPayloadBytes() > 0);
        assertTrue(storage.encodedMetadataBytes() > 0);

        float[] unwritten = {9, 9, 9, 9};
        storage.readKey(0, 1, unwritten, 0);
        assertArrayEquals(new float[4], unwritten);

        float[] rangeKeys = {
                0.2f, 0.1f, -0.3f, 0.5f,
                -0.6f, 0.4f, 0.7f, 0.1f
        };
        float[] rangeValues = {
                -0.1f, 0.2f, 0.6f, 0.3f,
                0.5f, -0.4f, 0.2f, 0.8f
        };
        storage.writeRange(0, 3, 2, rangeKeys, 0, rangeValues, 0);
        float[] decodedRange = new float[8];
        storage.readKeyRange(0, 3, 2, decodedRange, 0);
        assertFiniteAndNonZero(decodedRange);
        assertEquals(4, storage.storedPositionCount());
    }

    @Test
    void sessionVisibilityAndTruncationRemainOwnedBySession() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 4);
        CompressedKVCacheStorage storage =
                new CompressedKVCacheStorage(layout, new TurboQuantProdKVCacheCodec(3, 19L));
        KVCacheSession session = new KVCacheSession(storage);
        float[] key = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] value = {0.5f, -0.2f, 0.7f, 0.1f};

        session.write(0, 0, key, 0, value, 0);
        float[] current = new float[4];
        session.readKeyRangeForAttention(0, 0, 1, current, 0);
        assertFiniteAndNonZero(current);
        assertThrows(IndexOutOfBoundsException.class,
                () -> session.readKey(0, 0, new float[4], 0));

        session.commit(1);
        session.readKey(0, 0, current, 0);
        assertFiniteAndNonZero(current);

        session.write(0, 1, key, 0, value, 0);
        session.commit(2);
        assertEquals(2, storage.storedPositionCount());

        session.truncate(1);
        assertEquals(1, session.committedLength());
        assertEquals(1, storage.storedPositionCount());
        float[] clearedSuffix = {9, 9, 9, 9};
        storage.readKey(0, 1, clearedSuffix, 0);
        assertArrayEquals(new float[4], clearedSuffix);

        session.reset();
        assertEquals(0, session.committedLength());
        assertEquals(0, storage.storedPositionCount());
    }

    @Test
    void failedValueEncodingDoesNotReplacePreviouslyStoredKeyOrValue() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 4, 1, 4);
        FailingCodec codec = new FailingCodec();
        CompressedKVCacheStorage storage = new CompressedKVCacheStorage(layout, codec);
        float[] originalKey = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] originalValue = {0.5f, -0.2f, 0.7f, 0.1f};
        storage.write(0, 0, originalKey, 0, originalValue, 0);

        float[] expectedKey = new float[4];
        float[] expectedValue = new float[4];
        storage.readKey(0, 0, expectedKey, 0);
        storage.readValue(0, 0, expectedValue, 0);

        codec.failValueEncoding = true;
        assertThrows(IllegalStateException.class,
                () -> storage.write(0, 0,
                        new float[]{9, 8, 7, 6}, 0,
                        new float[]{5, 4, 3, 2}, 0));

        float[] actualKey = new float[4];
        float[] actualValue = new float[4];
        storage.readKey(0, 0, actualKey, 0);
        storage.readValue(0, 0, actualValue, 0);
        assertArrayEquals(expectedKey, actualKey);
        assertArrayEquals(expectedValue, actualValue);
    }

    @Test
    void clearLayerAndCloseReleaseEncodedPositions() {
        KVCacheLayout layout = KVCacheLayout.uniform(2, 4, 1, 4);
        CompressedKVCacheStorage storage =
                new CompressedKVCacheStorage(layout, new TurboQuantProdKVCacheCodec(3, 23L));
        float[] vector = {0.2f, 0.3f, -0.1f, 0.6f};
        storage.write(0, 0, vector, 0, vector, 0);
        storage.write(1, 0, vector, 0, vector, 0);

        storage.clearLayer(0);
        assertEquals(1, storage.storedPositionCount());
        storage.truncate(0);
        assertEquals(0, storage.storedPositionCount());

        storage.write(1, 1, vector, 0, vector, 0);
        storage.close();
        assertThrows(IllegalStateException.class,
                () -> storage.readKey(1, 1, new float[4], 0));
        assertThrows(IllegalStateException.class, storage::clear);
    }

    private static void assertFiniteAndNonZero(float[] values) {
        boolean nonZero = false;
        for (float value : values) {
            assertTrue(Float.isFinite(value));
            nonZero |= value != 0.0f;
        }
        assertTrue(nonZero);
    }

    private static final class FailingCodec implements KVCacheCompressionCodec {
        private final TurboQuantProdKVCacheCodec delegate = new TurboQuantProdKVCacheCodec(3, 101L);
        private boolean failValueEncoding;

        @Override
        public String id() {
            return delegate.id();
        }

        @Override
        public int version() {
            return delegate.version();
        }

        @Override
        public KVCacheEncodedBlock encode(
                float[] source, int sourceOffset, KVCacheCompressionContext context) {
            if (failValueEncoding && context.tensorKind() == KVCacheTensorKind.VALUE) {
                throw new IllegalStateException("simulated value encoding failure");
            }
            return delegate.encode(source, sourceOffset, context);
        }

        @Override
        public void decode(
                KVCacheEncodedBlock encoded,
                float[] destination,
                int destinationOffset,
                KVCacheCompressionContext context) {
            delegate.decode(encoded, destination, destinationOffset, context);
        }
    }
}
