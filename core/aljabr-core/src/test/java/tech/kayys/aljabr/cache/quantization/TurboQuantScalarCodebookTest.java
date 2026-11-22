package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantScalarCodebookTest {

    @Test
    void buildsExpectedNumberOfLevelsAndMonotonicCentroids() {
        for (int bits = 1; bits <= 8; bits++) {
            TurboQuantScalarCodebook codebook = TurboQuantScalarCodebook.forDimension(128, bits);
            double[] centroids = codebook.centroids();

            assertEquals(1 << bits, codebook.levels());
            assertEquals(bits, codebook.bits());
            for (int i = 1; i < centroids.length; i++) {
                assertTrue(centroids[i] > centroids[i - 1],
                        "centroids must be strictly increasing for bits=" + bits);
            }
        }
    }

    @Test
    void codebookIsSymmetricForTheSphericalCoordinateDistribution() {
        TurboQuantScalarCodebook codebook = TurboQuantScalarCodebook.forDimension(64, 4);
        double[] centroids = codebook.centroids();

        for (int i = 0; i < centroids.length; i++) {
            assertEquals(-centroids[centroids.length - 1 - i], centroids[i], 2.0e-3);
        }
    }

    @Test
    void quantizesAndReconstructsWithinSupport() {
        TurboQuantScalarCodebook codebook = TurboQuantScalarCodebook.forDimension(128, 4);
        double[] boundaries = codebook.boundaries();

        for (int i = 0; i < codebook.levels(); i++) {
            double probe = (boundaries[i] + boundaries[i + 1]) * 0.5;
            int index = codebook.quantize(probe);
            assertEquals(i, index);
            assertTrue(Double.isFinite(codebook.reconstruct(index)));
        }
    }

    @Test
    void saturatesValuesOutsideTheoreticalSupport() {
        TurboQuantScalarCodebook codebook = TurboQuantScalarCodebook.forDimension(64, 3);
        assertEquals(0, codebook.quantize(-1.0e6));
        assertEquals(codebook.levels() - 1, codebook.quantize(1.0e6));
    }

    @Test
    void returnsDefensiveCopiesOfCodebookArrays() {
        TurboQuantScalarCodebook codebook = TurboQuantScalarCodebook.forDimension(32, 3);
        double[] centroids = codebook.centroids();
        double[] boundaries = codebook.boundaries();
        centroids[0] = 1000;
        boundaries[1] = 1000;

        assertNotEquals(1000, codebook.centroids()[0]);
        assertNotEquals(1000, codebook.boundaries()[1]);
    }

    @Test
    void rejectsUnsupportedDimensionsBitWidthsAndNonFiniteInputs() {
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantScalarCodebook.forDimension(2, 2));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantScalarCodebook.forDimension(64, 0));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantScalarCodebook.forDimension(64, 9));

        TurboQuantScalarCodebook codebook = TurboQuantScalarCodebook.forDimension(64, 2);
        assertThrows(IllegalArgumentException.class, () -> codebook.quantize(Double.NaN));
        assertThrows(IndexOutOfBoundsException.class, () -> codebook.reconstruct(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> codebook.reconstruct(codebook.levels()));
    }
}
