package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TurboQuantProdKVCacheCodecQualityTest {

    private static final int TOKEN_COUNT = 32;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIMENSION = 64;
    private static final long SEED = 0x51A7E5L;

    @Test
    void encodedPayloadAndMetadataAreSmallerThanFloat32ForRepresentativeBlocks() {
        KVCacheCompressionContext context = new KVCacheCompressionContext(
                0, 0, TOKEN_COUNT, KV_HEADS, HEAD_DIMENSION, KVCacheTensorKind.KEY);
        float[] source = gaussianBlock(context.elementCount(), 0x71B4C2L);
        int rawBytes = Math.multiplyExact(source.length, Float.BYTES);

        for (int bits : new int[]{2, 4, 8}) {
            KVCacheEncodedBlock encoded = new TurboQuantProdKVCacheCodec(bits, SEED)
                    .encode(source, 0, context);
            int encodedBytes = encoded.payloadSizeBytes() + encoded.metadataSizeBytes();
            assertTrue(encodedBytes < rawBytes,
                    () -> bits + "-bit TurboQuant-prod block used " + encodedBytes
                            + " bytes, not smaller than " + rawBytes + " raw bytes");
        }
    }

    @Test
    void everySupportedWidthDecodesDeterministicallyAcrossCodecInstances() {
        KVCacheCompressionContext context = new KVCacheCompressionContext(
                4, 23, 3, 2, HEAD_DIMENSION, KVCacheTensorKind.VALUE);
        float[] source = gaussianBlock(context.elementCount(), 0x2A11L);

        for (int bits = 1; bits <= 8; bits++) {
            TurboQuantProdKVCacheCodec encoder = new TurboQuantProdKVCacheCodec(bits, SEED);
            TurboQuantProdKVCacheCodec decoder = new TurboQuantProdKVCacheCodec(bits, SEED);
            KVCacheEncodedBlock encoded = encoder.encode(source, 0, context);
            float[] first = new float[source.length];
            float[] second = new float[source.length];

            encoder.decode(encoded, first, 0, context);
            decoder.decode(encoded, second, 0, context);

            assertArrayEquals(first, second, 0.0f, bits + "-bit decode must be deterministic");
            for (float value : first) {
                assertTrue(Float.isFinite(value), bits + "-bit decode must remain finite");
            }
        }
    }

    private static float[] gaussianBlock(int elementCount, long seed) {
        Random random = new Random(seed);
        float[] values = new float[elementCount];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) random.nextGaussian();
        }
        return values;
    }
}
