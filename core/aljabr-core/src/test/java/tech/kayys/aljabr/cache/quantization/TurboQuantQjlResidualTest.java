package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantQjlResidualTest {

    @Test
    void encodesOneBitPerCoordinateAndPreservesNorm() {
        TurboQuantQjlResidual quantizer = new TurboQuantQjlResidual(5, 0x5EEDL);
        float[] source = {99, 3, -4, 0, 0, 0, 101};

        TurboQuantQjlResidual.EncodedResidual encoded = quantizer.encode(source, 1);

        assertEquals(5, encoded.dimension());
        assertEquals(0x5EEDL, encoded.seed());
        assertEquals(5.0, encoded.norm(), 1.0e-12);
        assertEquals(1, encoded.payloadSizeBytes());
        assertEquals(8, encoded.metadataSizeBytes());
        assertEquals(0, encoded.packedSigns()[0] & 0b1110_0000);
    }

    @Test
    void encodingAndDecodingAreDeterministicForTheSameSeed() {
        float[] residual = {0.25f, -1.5f, 2.0f, 0.75f, -0.125f, 3.0f};
        TurboQuantQjlResidual first = new TurboQuantQjlResidual(6, 77);
        TurboQuantQjlResidual second = new TurboQuantQjlResidual(6, 77);

        var encodedFirst = first.encode(residual, 0);
        var encodedSecond = second.encode(residual, 0);

        assertArrayEquals(encodedFirst.packedSigns(), encodedSecond.packedSigns());
        assertArrayEquals(first.decode(encodedFirst), second.decode(encodedSecond));
    }

    @Test
    void zeroResidualReconstructsExactlyAsZero() {
        TurboQuantQjlResidual quantizer = new TurboQuantQjlResidual(4, 9);

        var encoded = quantizer.encode(new float[4], 0);

        assertEquals(0.0, encoded.norm());
        assertArrayEquals(new float[4], quantizer.decode(encoded));
    }

    @Test
    void rejectsMismatchedProjectionAndInvalidInputs() {
        TurboQuantQjlResidual quantizer = new TurboQuantQjlResidual(4, 9);
        var encoded = quantizer.encode(new float[]{1, 2, 3, 4}, 0);

        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantQjlResidual(4, 10).decode(encoded));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantQjlResidual(5, 9).decode(encoded));
        assertThrows(IllegalArgumentException.class,
                () -> quantizer.encode(new float[]{1, Float.NaN, 3, 4}, 0));
        assertThrows(IndexOutOfBoundsException.class,
                () -> quantizer.encode(new float[4], 1));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboQuantQjlResidual(0, 9));
    }

    @Test
    void estimatesInnerProductsWithoutSystematicBiasAcrossSeeds() {
        float[] residual = {1.0f, -2.0f, 0.5f, 3.0f};
        float[] query = {0.25f, 1.5f, -0.75f, 2.0f};
        double expected = dot(residual, query);
        double estimates = 0.0;
        int trials = 1024;

        for (int seed = 0; seed < trials; seed++) {
            TurboQuantQjlResidual quantizer = new TurboQuantQjlResidual(4, seed);
            float[] reconstructed = quantizer.decode(quantizer.encode(residual, 0));
            estimates += dot(reconstructed, query);
        }

        double meanEstimate = estimates / trials;
        assertEquals(expected, meanEstimate, 0.12,
                "QJL inner-product estimates should converge to the original dot product");
    }

    @Test
    void returnedSignPayloadIsDefensivelyCopied() {
        TurboQuantQjlResidual quantizer = new TurboQuantQjlResidual(8, 123);
        var encoded = quantizer.encode(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, 0);
        byte[] exposed = encoded.packedSigns();
        exposed[0] ^= (byte) 0xFF;

        assertNotEquals(exposed[0], encoded.packedSigns()[0]);
    }

    private static double dot(float[] left, float[] right) {
        double result = 0.0;
        for (int i = 0; i < left.length; i++) {
            result += (double) left[i] * right[i];
        }
        return result;
    }
}
