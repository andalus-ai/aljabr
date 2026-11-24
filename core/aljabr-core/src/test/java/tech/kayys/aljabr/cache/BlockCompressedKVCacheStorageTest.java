package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.quantization.TurboQuantProdKVCacheCodec;

import static org.junit.jupiter.api.Assertions.*;

class BlockCompressedKVCacheStorageTest {

    @Test
    void batchesRangesAcrossBlocksAndPreservesUnwrittenZeroPositions() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 10, 1, 4);
        BlockCompressedKVCacheStorage storage = new BlockCompressedKVCacheStorage(
                layout, new TurboQuantProdKVCacheCodec(4, 31L), 4);

        float[] keys = vectors(6, 0.1f);
        float[] values = vectors(6, -0.2f);
        storage.writeRange(0, 1, 6, keys, 0, values, 0);

        assertEquals(6, storage.storedPositionCount());
        assertEquals(2, storage.storedBlockCount());

        float[] decoded = new float[24];
        storage.readKeyRange(0, 1, 6, decoded, 0);
        assertFiniteAndNonZero(decoded);

        float[] unwritten = {7, 7, 7, 7};
        storage.readKey(0, 0, unwritten, 0);
        assertArrayEquals(new float[4], unwritten);
    }

    @Test
    void blockMetadataIsAmortizedComparedWithPositionStorage() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 4);
        TurboQuantProdKVCacheCodec blockCodec = new TurboQuantProdKVCacheCodec(4, 41L);
        TurboQuantProdKVCacheCodec positionCodec = new TurboQuantProdKVCacheCodec(4, 41L);
        BlockCompressedKVCacheStorage blocks =
                new BlockCompressedKVCacheStorage(layout, blockCodec, 4);
        CompressedKVCacheStorage positions =
                new CompressedKVCacheStorage(layout, positionCodec);

        float[] keys = vectors(8, 0.15f);
        float[] values = vectors(8, -0.12f);
        blocks.writeRange(0, 0, 8, keys, 0, values, 0);
        positions.writeRange(0, 0, 8, keys, 0, values, 0);

        assertTrue(blocks.encodedMetadataBytes() < positions.encodedMetadataBytes(),
                () -> "block metadata=" + blocks.encodedMetadataBytes()
                        + ", position metadata=" + positions.encodedMetadataBytes());
        assertEquals(2, blocks.storedBlockCount());
        assertEquals(8, blocks.storedPositionCount());
    }

    @Test
    void rewritingAndTruncatingPartialBlockPreservesPrefixAndClearsSuffix() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 9, 1, 4);
        BlockCompressedKVCacheStorage storage = new BlockCompressedKVCacheStorage(
                layout, new TurboQuantProdKVCacheCodec(3, 53L), 4);
        float[] keys = vectors(8, 0.2f);
        float[] values = vectors(8, -0.1f);
        storage.writeRange(0, 0, 8, keys, 0, values, 0);

        float[] replacement = {9, 8, 7, 6};
        storage.write(0, 2, replacement, 0, replacement, 0);
        storage.truncate(5);

        assertEquals(5, storage.storedPositionCount());
        float[] decoded = new float[4];
        storage.readKey(0, 2, decoded, 0);
        assertFiniteAndNonZero(decoded);

        float[] cleared = {3, 3, 3, 3};
        storage.readKey(0, 5, cleared, 0);
        assertArrayEquals(new float[4], cleared);
        storage.readKey(0, 7, cleared, 0);
        assertArrayEquals(new float[4], cleared);
    }

    @Test
    void failedValueEncodingDoesNotPublishPartiallyEncodedBlock() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 4);
        FailingCodec codec = new FailingCodec();
        BlockCompressedKVCacheStorage storage =
                new BlockCompressedKVCacheStorage(layout, codec, 4);
        float[] key = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] value = {0.5f, -0.2f, 0.7f, 0.1f};
        codec.failValue = true;

        assertThrows(IllegalStateException.class,
                () -> storage.write(0, 0, key, 0, value, 0));
        assertEquals(0, storage.storedBlockCount());
        assertEquals(0, storage.storedPositionCount());
    }

    private static float[] vectors(int tokens, float start) {
        float[] values = new float[tokens * 4];
        for (int i = 0; i < values.length; i++) {
            values[i] = start + (i + 1) * 0.0375f;
        }
        return values;
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
        private final TurboQuantProdKVCacheCodec delegate =
                new TurboQuantProdKVCacheCodec(3, 67L);
        private boolean failValue;

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
            if (failValue && context.tensorKind() == KVCacheTensorKind.VALUE) {
                throw new IllegalStateException("simulated value encoding failure");
            }
            return delegate.encode(source, sourceOffset, context);
        }

        @Override
        public void decode(
                KVCacheEncodedBlock encoded, float[] destination, int destinationOffset,
                KVCacheCompressionContext context) {
            delegate.decode(encoded, destination, destinationOffset, context);
        }
    }
}
