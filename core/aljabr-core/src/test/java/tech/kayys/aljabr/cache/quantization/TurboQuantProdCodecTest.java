package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantProdCodecTest {

    @Test
    void encodesAndDecodesCombinedMseAndQjlBlock() {
        KVCacheCompressionContext context = context(2, 2, 4);
        float[] source = {
                0.25f, -0.5f, 1.0f, 2.0f,
                -1.0f, 0.75f, 0.5f, -0.25f,
                2.0f, 1.0f, -0.5f, 0.125f,
                -0.75f, -1.5f, 0.25f, 0.5f
        };
        TurboQuantProdCodec codec = new TurboQuantProdCodec(4, 0x5EEDL);

        KVCacheEncodedBlock encoded = codec.encode(source, 0, context);
        float[] decoded = new float[source.length];
        codec.decode(encoded, decoded, 0, context);

        assertEquals(codec.id(), encoded.codecId());
        assertEquals(source.length, encoded.elementCount());
        assertTrue(encoded.payloadSizeBytes() > 0);
        assertTrue(encoded.metadataSizeBytes() > 0);
        for (float value : decoded) {
            assertTrue(Float.isFinite(value));
        }
        assertFalse(java.util.Arrays.equals(source, decoded),
                "QJL reconstruction is an estimator, not lossless compression");
    }

    @Test
    void usesOneFewerMseBitThanTheTotalProductWidth() {
        KVCacheCompressionContext context = context(1, 1, 8);
        float[] source = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        TurboQuantProdCodec prod = new TurboQuantProdCodec(5, 12);
        TurboQuantMseCodec mse = new TurboQuantMseCodec(4, 12);

        KVCacheEncodedBlock prodBlock = prod.encode(source, 0, context);
        KVCacheEncodedBlock mseBlock = mse.encode(source, 0, context);

        assertEquals(5, prod.bits());
        assertTrue(prodBlock.payloadSizeBytes() > mseBlock.payloadSizeBytes());
    }

    @Test
    void supportsSourceAndDestinationOffsets() {
        KVCacheCompressionContext context = context(1, 2, 4);
        float[] source = {99, 99, 1, 2, 3, 4, -1, -2, -3, -4, 99};
        TurboQuantProdCodec codec = new TurboQuantProdCodec(3, 7);
        KVCacheEncodedBlock encoded = codec.encode(source, 2, context);
        float[] destination = new float[12];

        codec.decode(encoded, destination, 3, context);

        assertEquals(0.0f, destination[0]);
        assertEquals(0.0f, destination[2]);
        assertEquals(0.0f, destination[11]);
        for (int i = 3; i < 11; i++) {
            assertTrue(Float.isFinite(destination[i]));
        }
    }

    @Test
    void deterministicForTheSameSeedAndInput() {
        KVCacheCompressionContext context = context(2, 1, 4);
        float[] source = {1, 2, 3, 4, -1, 3, 2, 0};
        TurboQuantProdCodec first = new TurboQuantProdCodec(4, 99);
        TurboQuantProdCodec second = new TurboQuantProdCodec(4, 99);

        KVCacheEncodedBlock firstBlock = first.encode(source, 0, context);
        KVCacheEncodedBlock secondBlock = second.encode(source, 0, context);

        assertArrayEquals(firstBlock.payload(), secondBlock.payload());
        assertArrayEquals(firstBlock.metadata(), secondBlock.metadata());
    }

    @Test
    void rejectsWrongConfigurationGeometryAndCorruptPayloads() {
        KVCacheCompressionContext context = context(1, 1, 4);
        TurboQuantProdCodec codec = new TurboQuantProdCodec(4, 5);
        KVCacheEncodedBlock encoded = codec.encode(new float[]{1, 2, 3, 4}, 0, context);

        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantProdCodec(3, 5).decode(encoded, new float[4], 0, context));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantProdCodec(4, 6).decode(encoded, new float[4], 0, context));
        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(encoded, new float[8], 0, context(1, 2, 4)));

        byte[] corruptPayload = encoded.payload();
        corruptPayload[0] ^= (byte) 0xFF;
        KVCacheEncodedBlock badPayload = new KVCacheEncodedBlock(
                encoded.codecId(), encoded.codecVersion(), encoded.elementCount(),
                corruptPayload, encoded.metadata());
        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(badPayload, new float[4], 0, context));
    }

    @Test
    void leavesDestinationUnchangedWhenQjlPayloadIsCorrupt() {
        KVCacheCompressionContext context = context(1, 1, 4);
        TurboQuantProdCodec codec = new TurboQuantProdCodec(4, 5);
        KVCacheEncodedBlock encoded = codec.encode(new float[]{1, 2, 3, 4}, 0, context);
        byte[] corruptPayload = encoded.payload();
        corruptPayload[corruptPayload.length - 1] |= (byte) 0x80;
        KVCacheEncodedBlock badPayload = new KVCacheEncodedBlock(
                encoded.codecId(), encoded.codecVersion(), encoded.elementCount(),
                corruptPayload, encoded.metadata());
        float[] destination = {17, 17, 17, 17};

        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(badPayload, destination, 0, context));
        assertArrayEquals(new float[]{17, 17, 17, 17}, destination,
                "failed decoding must not partially publish MSE output");
    }

    @Test
    void validatesTotalBitWidth() {
        assertThrows(IllegalArgumentException.class, () -> new TurboQuantProdCodec(1, 0));
        assertThrows(IllegalArgumentException.class, () -> new TurboQuantProdCodec(10, 0));
    }

    private static KVCacheCompressionContext context(int tokens, int heads, int dimension) {
        return new KVCacheCompressionContext(0, 0, tokens, heads, dimension, KVCacheTensorKind.KEY);
    }
}
