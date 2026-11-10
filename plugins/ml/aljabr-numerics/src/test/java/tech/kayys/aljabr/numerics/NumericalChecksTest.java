package tech.kayys.aljabr.numerics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NumericalChecksTest {
    @Test
    void rejectsNonFiniteValuesByDefault() {
        assertThrows(IllegalArgumentException.class,
                () -> NumericalChecks.requireFinite(Double.NaN, "sample"));
        assertEquals(Double.POSITIVE_INFINITY,
                NumericalChecks.requireFinite(Double.POSITIVE_INFINITY, "sample", NonFinitePolicy.ALLOW));
    }

    @Test
    void requiresStrictlyPositiveFiniteValues() {
        assertEquals(2.5, NumericalChecks.requirePositiveFinite(2.5, "scale"));
        assertThrows(IllegalArgumentException.class,
                () -> NumericalChecks.requirePositiveFinite(0.0, "scale"));
        assertThrows(IllegalArgumentException.class,
                () -> NumericalChecks.requirePositiveFinite(Double.POSITIVE_INFINITY, "scale"));
    }
}
