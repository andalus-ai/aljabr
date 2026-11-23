package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TurboQuantMseKVCacheCodecQualityTest {

    private static final int TOKEN_COUNT = 32;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIMENSION = 64;
    private static final long TRANSFORM_SEED = 0x51A7E5L;

    @Test
    void reconstructionQualityImprovesWithBitWidthAndRetainsUsefulCompression() {
        KVCacheCompressionContext context = new KVCacheCompressionContext(
                0, 0, TOKEN_COUNT, KV_HEADS, HEAD_DIMENSION, KVCacheTensorKind.KEY);
        float[] source = gaussianBlock(context.elementCount(), 0x71B4C2L);

        double twoBitCosine = evaluate(source, context, 2);
        double fourBitCosine = evaluate(source, context, 4);
        double eightBitCosine = evaluate(source, context, 8);

        assertTrue(twoBitCosine > 0.70,
                () -> "2-bit mean cosine similarity was " + twoBitCosine);
        assertTrue(fourBitCosine > twoBitCosine,
                () -> "4-bit quality should improve over 2-bit: " + fourBitCosine + " <= " + twoBitCosine);
        assertTrue(eightBitCosine > fourBitCosine,
                () -> "8-bit quality should improve over 4-bit: " + eightBitCosine + " <= " + fourBitCosine);
    }

    @Test
    void encodedPayloadAndMetadataAreSmallerThanFloat32ForRepresentativeBlocks() {
        KVCacheCompressionContext context = new KVCacheCompressionContext(
                0, 0, TOKEN_COUNT, KV_HEADS, HEAD_DIMENSION, KVCacheTensorKind.VALUE);
        float[] source = gaussianBlock(context.elementCount(), 0x71B4C2L);
        int rawFloat32Bytes = Math.multiplyExact(source.length, Float.BYTES);

        for (int bits : new int[]{2, 4, 8}) {
            KVCacheEncodedBlock encoded =
                    new TurboQuantMseKVCacheCodec(bits, TRANSFORM_SEED).encode(source, 0, context);
            int encodedBytes = encoded.payloadSizeBytes() + encoded.metadataSizeBytes();

            assertTrue(encodedBytes < rawFloat32Bytes,
                    () -> bits + "-bit encoded block used " + encodedBytes
                            + " bytes, not smaller than " + rawFloat32Bytes + " raw bytes");
        }
    }

    private static double evaluate(float[] source, KVCacheCompressionContext context, int bits) {
        TurboQuantMseKVCacheCodec codec = new TurboQuantMseKVCacheCodec(bits, TRANSFORM_SEED);
        KVCacheEncodedBlock encoded = codec.encode(source, 0, context);
        float[] reconstructed = new float[source.length];
        codec.decode(encoded, reconstructed, 0, context);

        double cosineSum = 0.0;
        int vectorCount = context.tokenCount() * context.kvHeads();
        for (int vector = 0; vector < vectorCount; vector++) {
            int offset = vector * context.headDimension();
            double dot = 0.0;
            double sourceSquaredNorm = 0.0;
            double reconstructedSquaredNorm = 0.0;
            for (int i = 0; i < context.headDimension(); i++) {
                double original = source[offset + i];
                double actual = reconstructed[offset + i];
                dot += original * actual;
                sourceSquaredNorm += original * original;
                reconstructedSquaredNorm += actual * actual;
            }
            cosineSum += dot / Math.sqrt(sourceSquaredNorm * reconstructedSquaredNorm);
        }
        return cosineSum / vectorCount;
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
