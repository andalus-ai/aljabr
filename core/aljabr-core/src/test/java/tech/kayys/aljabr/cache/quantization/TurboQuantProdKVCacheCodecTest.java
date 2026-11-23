package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantProdKVCacheCodecTest {

    @Test
    void combinesMseAndQjlForMultiTokenMultiHeadBlocks() {
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(2, 19, 2, 2, 5, KVCacheTensorKind.KEY);
        float[] source = new float[context.elementCount() + 2];
        source[0] = 91.0f;
        source[source.length - 1] = 92.0f;
        for (int i = 0; i < context.elementCount(); i++) {
            source[i + 1] = (float) (Math.sin(i * 0.37) * 0.8 + Math.cos(i * 0.11) * 0.2);
        }
        float[] original = new float[context.elementCount()];
        System.arraycopy(source, 1, original, 0, original.length);

        float[] sourceBefore = source.clone();
        TurboQuantProdKVCacheCodec codec = new TurboQuantProdKVCacheCodec(4, 0x1234ABCDL);
        KVCacheEncodedBlock encoded = codec.encode(source, 1, context);
        float[] destination = new float[context.elementCount() + 4];
        destination[0] = 77.0f;
        destination[1] = 78.0f;
        destination[destination.length - 1] = 79.0f;

        codec.decode(encoded, destination, 2, context);

        assertEquals(TurboQuantProdKVCacheCodec.CODEC_ID, encoded.codecId());
        assertEquals(TurboQuantProdKVCacheCodec.CODEC_VERSION, encoded.codecVersion());
        assertEquals(context.elementCount(), encoded.elementCount());
        assertTrue(encoded.payloadSizeBytes() > 0);
        assertTrue(encoded.metadataSizeBytes() > 52);
        assertEquals(77.0f, destination[0]);
        assertEquals(78.0f, destination[1]);
        assertEquals(79.0f, destination[destination.length - 1]);
        assertArrayEquals(sourceBefore, source);

        for (int i = 0; i < original.length; i++) {
            assertTrue(Float.isFinite(destination[i + 2]));
        }
    }

    @Test
    void oneBitConfigurationUsesPureQjlWithoutMsePayload() {
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(0, 0, 1, 2, 5, KVCacheTensorKind.VALUE);
        float[] source = {0.2f, -0.4f, 0.7f, 0.1f, -0.3f, -0.5f, 0.3f, 0.2f, 0.8f, -0.1f};
        TurboQuantProdKVCacheCodec codec = new TurboQuantProdKVCacheCodec(1, 42L);

        KVCacheEncodedBlock encoded = codec.encode(source, 0, context);
        float[] decoded = new float[source.length];
        codec.decode(encoded, decoded, 0, context);

        // Five sign bits are rounded to one byte per vector; no MSE indices are present.
        assertEquals(2, encoded.payloadSizeBytes());
        assertEquals(56 + 2 * Double.BYTES, encoded.metadataSizeBytes());
        for (float value : decoded) {
            assertTrue(Float.isFinite(value));
        }
    }

    @Test
    void zeroResidualAndZeroInputDecodeSafely() {
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(0, 0, 1, 1, 4, KVCacheTensorKind.KEY);
        TurboQuantProdKVCacheCodec codec = new TurboQuantProdKVCacheCodec(3, 7L);
        float[] source = {0, 0, 0, 0};

        KVCacheEncodedBlock encoded = codec.encode(source, 0, context);
        float[] decoded = {9, 9, 9, 9};
        codec.decode(encoded, decoded, 0, context);

        assertArrayEquals(source, decoded);
    }

    @Test
    void rejectsConfigurationAndGeometryMismatchesWithoutPublishingPartialOutput() {
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(1, 4, 1, 1, 4, KVCacheTensorKind.KEY);
        TurboQuantProdKVCacheCodec codec = new TurboQuantProdKVCacheCodec(4, 11L);
        KVCacheEncodedBlock encoded = codec.encode(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, 0, context);

        float[] destination = {8, 8, 8, 8};
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantProdKVCacheCodec(3, 11L).decode(encoded, destination, 0, context));
        assertArrayEquals(new float[]{8, 8, 8, 8}, destination);

        byte[] corruptedMetadata = encoded.metadata();
        corruptedMetadata[0] ^= 0x01;
        KVCacheEncodedBlock corrupted = new KVCacheEncodedBlock(
                encoded.codecId(), encoded.codecVersion(), encoded.elementCount(),
                encoded.payload(), corruptedMetadata);
        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(corrupted, destination, 0, context));
        assertArrayEquals(new float[]{8, 8, 8, 8}, destination);

        KVCacheCompressionContext wrongPosition =
                new KVCacheCompressionContext(1, 5, 1, 1, 4, KVCacheTensorKind.KEY);
        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(encoded, destination, 0, wrongPosition));
    }

    @Test
    void pureQjlRejectsKeyValueTensorKindMismatch() {
        KVCacheCompressionContext keyContext =
                new KVCacheCompressionContext(0, 3, 1, 1, 5, KVCacheTensorKind.KEY);
        KVCacheCompressionContext valueContext =
                new KVCacheCompressionContext(0, 3, 1, 1, 5, KVCacheTensorKind.VALUE);
        TurboQuantProdKVCacheCodec codec = new TurboQuantProdKVCacheCodec(1, 19L);
        KVCacheEncodedBlock encoded = codec.encode(new float[]{1, 2, 3, 4, 5}, 0, keyContext);

        float[] destination = {9, 9, 9, 9, 9};
        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(encoded, destination, 0, valueContext));
        assertArrayEquals(new float[]{9, 9, 9, 9, 9}, destination);
    }

    @Test
    void rejectsInvalidBitWidthsAndNonFiniteSource() {
        assertThrows(IllegalArgumentException.class, () -> new TurboQuantProdKVCacheCodec(0, 1L));
        assertThrows(IllegalArgumentException.class, () -> new TurboQuantProdKVCacheCodec(9, 1L));

        KVCacheCompressionContext context =
                new KVCacheCompressionContext(0, 0, 1, 1, 4, KVCacheTensorKind.VALUE);
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantProdKVCacheCodec(1, 1L)
                        .encode(new float[]{1, 2, Float.NaN, 4}, 0, context));
    }
}
