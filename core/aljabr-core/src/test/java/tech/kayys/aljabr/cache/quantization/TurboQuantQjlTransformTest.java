package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TurboQuantQjlTransformTest {

    @Test
    void seededProjectionProducesDeterministicPackedSignsAndReconstruction() {
        float[] residual = {0.25f, -0.5f, 1.0f, 0.75f, -0.125f};
        TurboQuantQjlTransform first = new TurboQuantQjlTransform(5, 91L);
        TurboQuantQjlTransform second = new TurboQuantQjlTransform(5, 91L);

        byte[] firstSigns = first.quantize(residual, 0);
        byte[] secondSigns = second.quantize(residual, 0);
        assertArrayEquals(firstSigns, secondSigns);
        assertEquals(first.residualNorm(residual, 0), second.residualNorm(residual, 0));

        float[] firstDecoded = new float[5];
        float[] secondDecoded = new float[5];
        first.reconstruct(firstSigns, first.residualNorm(residual, 0), firstDecoded, 0);
        second.reconstruct(secondSigns, second.residualNorm(residual, 0), secondDecoded, 0);
        assertArrayEquals(firstDecoded, secondDecoded);
    }

    @Test
    void reconstructionUsesOnlyOnePackedBitPerCoordinate() {
        TurboQuantQjlTransform transform = new TurboQuantQjlTransform(65, 7L);
        byte[] signs = transform.quantize(new float[65], 0);
        assertEquals(9, signs.length);
    }

    @Test
    void zeroResidualReconstructsToZero() {
        TurboQuantQjlTransform transform = new TurboQuantQjlTransform(8, 17L);
        float[] destination = new float[8];
        transform.reconstruct(transform.quantize(new float[8], 0), 0.0, destination, 0);
        assertArrayEquals(new float[8], destination);
    }

    @Test
    void qjlInnerProductEstimatorIsUnbiasedAcrossIndependentSeeds() {
        int dimension = 16;
        float[] residual = new float[dimension];
        float[] query = new float[dimension];
        SplittableRandom random = new SplittableRandom(0x71A5L);
        for (int i = 0; i < dimension; i++) {
            residual[i] = (float) random.nextDouble(-1.0, 1.0);
            query[i] = (float) random.nextDouble(-1.0, 1.0);
        }

        double residualNorm = new TurboQuantQjlTransform(dimension, 1L).residualNorm(residual, 0);
        double expected = dot(query, residual);
        double estimateSum = 0.0;
        int trials = 4096;
        for (int trial = 0; trial < trials; trial++) {
            TurboQuantQjlTransform transform = new TurboQuantQjlTransform(dimension, trial + 100L);
            byte[] signs = transform.quantize(residual, 0);
            float[] decoded = new float[dimension];
            transform.reconstruct(signs, residualNorm, decoded, 0);
            estimateSum += dot(query, decoded);
        }

        double meanEstimate = estimateSum / trials;
        assertTrue(Math.abs(meanEstimate - expected) < 0.08,
                () -> "QJL mean estimate " + meanEstimate + " differs from exact inner product " + expected);
    }

    @Test
    void rejectsInvalidInputsAndLeavesDestinationUnchangedOnFailure() {
        TurboQuantQjlTransform transform = new TurboQuantQjlTransform(3, 1L);
        assertThrows(IllegalArgumentException.class, () -> new TurboQuantQjlTransform(0, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> transform.quantize(new float[]{1.0f, Float.NaN, 2.0f}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> transform.reconstruct(new byte[]{0, 0}, 1.0, new float[3], 0));

        float[] destination = {4.0f, 5.0f, 6.0f};
        assertThrows(IllegalArgumentException.class,
                () -> transform.reconstruct(new byte[]{0}, Double.NaN, destination, 0));
        assertArrayEquals(new float[]{4.0f, 5.0f, 6.0f}, destination);
    }

    private static double dot(float[] left, float[] right) {
        double result = 0.0;
        for (int i = 0; i < left.length; i++) {
            result += (double) left[i] * right[i];
        }
        return result;
    }
}
