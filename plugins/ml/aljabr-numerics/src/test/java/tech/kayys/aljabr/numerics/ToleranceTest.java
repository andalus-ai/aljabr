package tech.kayys.aljabr.numerics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ToleranceTest {
    @Test
    void combinesAbsoluteAndRelativeTolerance() {
        Tolerance tolerance = new Tolerance(1.0e-8, 1.0e-6);
        assertTrue(tolerance.isClose(1_000_000.0, 1_000_000.5));
        assertTrue(tolerance.isClose(0.0, 5.0e-9));
        assertFalse(tolerance.isClose(0.0, 2.0e-8));
    }

    @Test
    void handlesNonFiniteValuesDeterministically() {
        Tolerance tolerance = new Tolerance(0.0, 0.0);
        assertTrue(tolerance.isClose(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
        assertFalse(tolerance.isClose(Double.NaN, Double.NaN));
    }

    @Test
    void rejectsInvalidTolerance() {
        assertThrows(IllegalArgumentException.class, () -> new Tolerance(-1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new Tolerance(0.0, Double.NaN));
    }
}
