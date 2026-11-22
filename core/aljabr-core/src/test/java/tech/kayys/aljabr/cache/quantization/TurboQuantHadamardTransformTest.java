package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantHadamardTransformTest {

    @Test
    void forwardAndInverseRoundTripAcrossPowerOfTwoDimensions() {
        for (int dimension : new int[] {1, 2, 4, 8, 16, 64, 128}) {
            TurboQuantHadamardTransform transform =
                    new TurboQuantHadamardTransform(dimension, 0x5EEDL);
            float[] source = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                source[i] = (float) (Math.sin(i * 0.37) * 3.0 + Math.cos(i * 0.11));
            }

            float[] encoded = new float[dimension];
            float[] decoded = new float[dimension];
            transform.forward(source, 0, encoded, 0);
            transform.inverse(encoded, 0, decoded, 0);

            for (int i = 0; i < dimension; i++) {
                assertEquals(source[i], decoded[i], 2.0e-5f,
                        "dimension=" + dimension + ", index=" + i);
            }
        }
    }

    @Test
    void transformIsDeterministicForTheSameSeed() {
        float[] source = {1, -2, 3, -4, 5, -6, 7, -8};
        float[] first = new float[8];
        float[] second = new float[8];

        new TurboQuantHadamardTransform(8, 42L).forward(source, 0, first, 0);
        new TurboQuantHadamardTransform(8, 42L).forward(source, 0, second, 0);

        assertArrayEquals(first, second);
    }

    @Test
    void differentSeedsProduceDifferentMixingTransforms() {
        float[] source = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] first = new float[8];
        float[] second = new float[8];

        new TurboQuantHadamardTransform(8, 1L).forward(source, 0, first, 0);
        new TurboQuantHadamardTransform(8, 2L).forward(source, 0, second, 0);

        assertFalse(java.util.Arrays.equals(first, second));
    }

    @Test
    void transformPreservesEuclideanNormWithinFloatingPointTolerance() {
        TurboQuantHadamardTransform transform = new TurboQuantHadamardTransform(16, 7L);
        float[] source = new float[16];
        for (int i = 0; i < source.length; i++) {
            source[i] = i - 5.5f;
        }
        float[] transformed = new float[16];

        transform.forward(source, 0, transformed, 0);

        assertEquals(normSquared(source), normSquared(transformed), 1.0e-4);
    }

    @Test
    void supportsOffsetsAndInPlaceOperation() {
        TurboQuantHadamardTransform transform = new TurboQuantHadamardTransform(4, 11L);
        float[] values = {99, 1, 2, 3, 4, 88};
        float[] expected = new float[4];
        transform.forward(values, 1, expected, 0);
        transform.forward(values, 1, values, 1);
        assertArrayEquals(expected, new float[] {values[1], values[2], values[3], values[4]});
    }

    @Test
    void rejectsInvalidDimensionsRangesAndNonFiniteValues() {
        assertThrows(IllegalArgumentException.class, () -> new TurboQuantHadamardTransform(0, 1));
        assertThrows(IllegalArgumentException.class, () -> new TurboQuantHadamardTransform(3, 1));

        TurboQuantHadamardTransform transform = new TurboQuantHadamardTransform(4, 1);
        assertThrows(IndexOutOfBoundsException.class,
                () -> transform.forward(new float[4], 1, new float[4], 0));
        assertThrows(IndexOutOfBoundsException.class,
                () -> transform.inverse(new float[4], 0, new float[4], 1));
        assertThrows(IllegalArgumentException.class,
                () -> transform.forward(new float[] {1, 2, Float.NaN, 4}, 0, new float[4], 0));
        assertThrows(NullPointerException.class,
                () -> transform.forward(null, 0, new float[4], 0));
    }

    private static double normSquared(float[] values) {
        double result = 0;
        for (float value : values) {
            result += (double) value * value;
        }
        return result;
    }
}
