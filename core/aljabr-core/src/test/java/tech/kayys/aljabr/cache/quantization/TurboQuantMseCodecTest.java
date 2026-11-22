package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantMseCodecTest {

    @Test
    void encodesAndDecodesPaddedMultiTokenMultiHeadBlock() {
        KVCacheCompressionContext context = context(3, 2, 5);
        float[] source = new float[context.elementCount() + 4];
        for (int i = 0; i < context.elementCount(); i++) {
            source[i + 2] = (float) (Math.sin(i * 0.37) * (1.0 + (i % 5)));
        }
        TurboQuantMseCodec codec = new TurboQuantMseCodec(8, 0x5EEDL);

        KVCacheEncodedBlock encoded = codec.encode(source, 2, context);
        float[] decoded = new float[context.elementCount() + 6];
        codec.decode(encoded, decoded, 3, context);

        assertEquals(context.elementCount(), encoded.elementCount());
        assertEquals(TurboQuantMseCodec.CODEC_ID, encoded.codecId());
        assertEquals(TurboQuantMseCodec.CODEC_VERSION, encoded.codecVersion());
        assertTrue(encoded.payloadSizeBytes() < context.elementCount() * Float.BYTES);
        assertEquals(32 + 6 * Double.BYTES, encoded.metadataSizeBytes());

        double squaredError = 0.0;
        double signalEnergy = 0.0;
        for (int i = 0; i < context.elementCount(); i++) {
            double expected = source[i + 2];
            double actual = decoded[i + 3];
            assertTrue(Double.isFinite(actual));
            squaredError += (expected - actual) * (expected - actual);
            signalEnergy += expected * expected;
        }
        assertTrue(squaredError / signalEnergy < 0.02,
                "relative squared error should remain small for the 8-bit reference path");
    }

    @Test
    void preservesZeroVectorsAndVectorScale() {
        KVCacheCompressionContext context = context(2, 1, 4);
        float[] source = {0, 0, 0, 0, 3, 4, 0, 0};
        TurboQuantMseCodec codec = new TurboQuantMseCodec(6, 42);

        KVCacheEncodedBlock encoded = codec.encode(source, 0, context);
        float[] decoded = new float[source.length];
        codec.decode(encoded, decoded, 0, context);

        for (int i = 0; i < 4; i++) {
            assertEquals(0.0f, decoded[i]);
        }
        double originalNorm = 5.0;
        double decodedNorm = 0.0;
        for (int i = 4; i < 8; i++) {
            decodedNorm += decoded[i] * decoded[i];
        }
        assertEquals(originalNorm, Math.sqrt(decodedNorm), 1.0e-4);
    }

    @Test
    void rejectsWrongCodecConfigurationAndContextGeometry() {
        KVCacheCompressionContext context = context(1, 1, 4);
        float[] source = {1, 2, 3, 4};
        TurboQuantMseCodec codec = new TurboQuantMseCodec(4, 9);
        KVCacheEncodedBlock encoded = codec.encode(source, 0, context);

        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseCodec(3, 9).decode(encoded, new float[4], 0, context));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantMseCodec(4, 10).decode(encoded, new float[4], 0, context));
        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(encoded, new float[8], 0, context(1, 1, 8)));
    }

    @Test
    void rejectsCorruptMetadataAndNonFiniteSourceValues() {
        KVCacheCompressionContext context = context(1, 1, 4);
        TurboQuantMseCodec codec = new TurboQuantMseCodec(4, 9);
        KVCacheEncodedBlock encoded = codec.encode(new float[]{1, 2, 3, 4}, 0, context);

        byte[] corruptMetadata = encoded.metadata();
        corruptMetadata[0] ^= 0x7F;
        KVCacheEncodedBlock corrupt = new KVCacheEncodedBlock(
                encoded.codecId(), encoded.codecVersion(), encoded.elementCount(),
                encoded.payload(), corruptMetadata);
        assertThrows(IllegalArgumentException.class,
                () -> codec.decode(corrupt, new float[4], 0, context));
        assertThrows(IllegalArgumentException.class,
                () -> codec.encode(new float[]{1, Float.NaN, 3, 4}, 0, context));
    }

    @Test
    void rejectsInvalidSourceAndDestinationRanges() {
        KVCacheCompressionContext context = context(1, 1, 4);
        TurboQuantMseCodec codec = new TurboQuantMseCodec(4, 9);
        assertThrows(IndexOutOfBoundsException.class,
                () -> codec.encode(new float[4], 1, context));
        KVCacheEncodedBlock encoded = codec.encode(new float[4], 0, context);
        assertThrows(IndexOutOfBoundsException.class,
                () -> codec.decode(encoded, new float[4], 1, context));
    }

    private static KVCacheCompressionContext context(int tokens, int heads, int dimension) {
        return new KVCacheCompressionContext(
                2, 16, tokens, heads, dimension, KVCacheTensorKind.KEY);
    }
}
