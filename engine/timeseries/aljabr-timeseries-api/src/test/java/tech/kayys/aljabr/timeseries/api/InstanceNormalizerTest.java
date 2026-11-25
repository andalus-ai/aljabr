package tech.kayys.aljabr.timeseries.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class InstanceNormalizerTest {
    @Test
    void standardizesObservedValuesAndPreservesMissingValues() {
        var result = new InstanceNormalizer().normalize(
                new double[] {1.0, 2.0, Double.NaN, 3.0});
        assertEquals(2.0, result.state().location(), 1e-12);
        assertEquals(Math.sqrt(2.0 / 3.0), result.state().scale(), 1e-12);
        assertFalse(result.observedMask()[2]);
        assertEquals(Double.NaN, result.values()[2]);
    }

    @Test
    void inverseRestoresOriginalUnitsAndArcsinhTransform() {
        var result = new InstanceNormalizer(1e-5, true).normalize(
                new double[] {-10.0, 0.0, 10.0});
        double[] restored = result.state().inverse(result.values());
        assertArrayEquals(new double[] {-10.0, 0.0, 10.0}, restored, 1e-10);
    }

    @Test
    void constantSeriesUsesConfiguredEpsilon() {
        var result = new InstanceNormalizer(1e-4, false).normalize(new double[] {7.0, 7.0});
        assertEquals(1e-4, result.state().scale(), 0.0);
        assertArrayEquals(new double[] {0.0, 0.0}, result.values(), 0.0);
    }

    @Test
    void allMissingSeriesHasFiniteFallbackStatistics() {
        var result = new InstanceNormalizer().normalize(new double[] {Double.NaN, Double.NaN});
        assertEquals(0.0, result.state().location());
        assertEquals(1.0, result.state().scale());
        assertFalse(result.observedMask()[0]);
    }

    @Test
    void rejectsObservedNanAndMismatchedMask() {
        assertThrows(IllegalArgumentException.class,
                () -> new InstanceNormalizer().normalize(new double[] {Double.NaN}, new boolean[] {true}));
        assertThrows(IllegalArgumentException.class,
                () -> new InstanceNormalizer().normalize(new double[] {1.0}, new boolean[0]));
    }
}
