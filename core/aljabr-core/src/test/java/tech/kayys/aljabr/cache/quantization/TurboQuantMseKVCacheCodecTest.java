package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantMseKVCacheCodecTest {

    @Test
    void encodesAndDecodesPaddedMultiTokenMultiHeadBlocks() {
        TurboQuantMseKVCacheCodec codec = new TurboQuantMseKVCacheCodec(4, 0x1234ABCDL);
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(3, 17, 2, 2, 5, KVCacheTensorKind.KEY);
        float[] source = {
                99, 98,
                0.2f, -0.4f, 0.7f, 0.1f, -0.3f,
                -0.5f, 0.3f, 0.2f, 0.8f, -0.1f,
                0.6f, 0.1f, -0.2f, 0.4f, 0.3f,
                -0.1f, -0.7f, 0.5f, 0.2f, 0.4f,
                97, 96
        };
        float[] original = new float[context.elementCount()];
        System.arraycopy(source, 2, original, 0, original.length);

        KVCacheEncodedBlock encoded = codec.encode(source, 2, context);
        float[] destination = new float[context.elementCount() + 4];
        destination[0] = 123;
        destination[1] = 123;
        destination[destination.length - 1] = 123;

        codec.decode(encoded, destination, 2, context);

        assertEquals(TurboQuantMseKVCacheCodec.CODEC_ID, encoded.codecId());
        assertEquals(TurboQuantMseKVCacheCodec.CODEC_VERSION, encoded.codecVersion());
        assertEquals(context.elementCount(), encoded.elementCount());
        assertTrue(encoded.payloadSizeBytes() < original.length * Float.BYTES);
        assertEquals(123, destination[0]);
        assertEquals(123, destination[1]);
        assertEquals(123, destination[destination.length - 1]);

        for (int vector = 0; vector < context.tokenCount() * context.kvHeads(); vector++) {
            double dot = 0.0;
            double originalNorm = 0.0;
            double decodedNorm = 0.0;
            for (int i = 0; i < context.headDimension(); i++) {
                int index = vector * context.headDimension() + i;
                float actual = destination[2 + index];
                assertTrue(Float.isFinite(actual));
                dot += original[index] * actual;
                originalNorm += (double) original[index] * original[index];
                decodedNorm += (double) actual * actual;
            }
            assertTrue(dot > 0.0, "decoded vector should preserve direction");
            assertTrue(decodedNorm > 0.0);
            assertTrue(Math.sqrt(decodedNorm / originalNorm) > 0.5);
            assertTrue(Math.sqrt(decodedNorm / originalNorm) < 1.5);
        }
    }

    @Test
    void preservesZeroVectorsAndDoesNotMutateInput() {
        TurboQuantMseKVCacheCodec codec = new TurboQuantMseKVCacheCodec(3, 42L);
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(0, 0, 1, 1, 7, KVCacheTensorKind.VALUE);
        float[] source = {0, 0, 0, 0, 0, 0, 0};
        float[] before = source.clone();

        KVCacheEncodedBlock encoded = codec.encode(source, 0, context);
        float[] decoded = new float[source.length];
        codec.decode(encoded, decoded, 0, context);

        assertArrayEquals(before, source);
        assertArrayEquals(before, decoded);
    }

    @Test
    void encodedMetadataMakesDecodeIndependentOfCodecInstanceState() {
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(1, 9, 1, 1, 4, KVCacheTensorKind.KEY);
        float[] source = {0.2f, 0.4f, -0.1f, 0.7f};

        KVCacheEncodedBlock encoded =
                new TurboQuantMseKVCacheCodec(5, 77L).encode(source, 0, context);
        float[] decoded = new float[source.length];

        new TurboQuantMseKVCacheCodec(5, 77L).decode(encoded, decoded, 0, context);
        assertTrue(Float.isFinite(decoded[0]));
        assertTrue(Float.isFinite(decoded[1]));
    }

    @Test
    void rejectsMismatchedCodecConfigurationAndGeometry() {
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(2, 11, 1, 1, 4, KVCacheTensorKind.KEY);
        float[] source = {0.1f, 0.2f, 0.3f, 0.4f};
        KVCacheEncodedBlock encoded =
                new TurboQuantMseKVCacheCodec(4, 10L).encode(source, 0, context);

        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseKVCacheCodec(3, 10L).decode(encoded, new float[4], 0, context));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseKVCacheCodec(4, 11L).decode(encoded, new float[4], 0, context));

        KVCacheCompressionContext wrongLayer =
                new KVCacheCompressionContext(3, 11, 1, 1, 4, KVCacheTensorKind.KEY);
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseKVCacheCodec(4, 10L).decode(encoded, new float[4], 0, wrongLayer));
    }

    @Test
    void failedDecodeDoesNotPartiallyModifyDestination() {
        TurboQuantMseKVCacheCodec codec = new TurboQuantMseKVCacheCodec(4, 10L);
        KVCacheCompressionContext context =
                new KVCacheCompressionContext(0, 0, 1, 1, 4, KVCacheTensorKind.KEY);
        KVCacheEncodedBlock encoded = codec.encode(new float[]{1, 2, 3, 4}, 0, context);
        byte[] invalidMetadata = encoded.metadata();
        invalidMetadata[0] ^= 0x01;
        KVCacheEncodedBlock corrupted = new KVCacheEncodedBlock(
                encoded.codecId(), encoded.codecVersion(), encoded.elementCount(),
                encoded.payload(), invalidMetadata);
        float[] destination = {9, 9, 9, 9};

        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(corrupted, destination, 0, context));
        assertArrayEquals(new float[]{9, 9, 9, 9}, destination);
    }

    @Test
    void rejectsInvalidInputAndUnsupportedHeadDimensions() {
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseKVCacheCodec(0, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseKVCacheCodec(9, 1L));

        KVCacheCompressionContext tooSmall =
                new KVCacheCompressionContext(0, 0, 1, 1, 2, KVCacheTensorKind.KEY);
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseKVCacheCodec(4, 1L).encode(new float[]{1, 2}, 0, tooSmall));

        KVCacheCompressionContext normal =
                new KVCacheCompressionContext(0, 0, 1, 1, 4, KVCacheTensorKind.KEY);
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseKVCacheCodec(4, 1L).encode(
                        new float[]{1, Float.NaN, 3, 4}, 0, normal));
    }
}
