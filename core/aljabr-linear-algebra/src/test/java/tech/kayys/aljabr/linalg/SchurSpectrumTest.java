package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SchurSpectrumTest {
    @Test void extractsComplexConjugatePairAndStabilityDiagnostics() {
        Matrix rotation = Matrix.of(new double[][]{{0, -2}, {2, 0}});
        SchurSpectrum spectrum = SchurSpectrum.of(RealSchurDecomposition.factor(rotation));
        assertEquals(2, spectrum.eigenvalues().size());
        assertEquals(0.0, spectrum.eigenvalues().get(0).real(), 1e-9);
        assertEquals(2.0, Math.abs(spectrum.eigenvalues().get(0).imaginary()), 1e-9);
        assertEquals(2.0, spectrum.spectralRadius(), 1e-9);
        assertEquals(0.0, spectrum.spectralAbscissa(), 1e-9);
        assertEquals(4.0, spectrum.determinant(), 1e-9);
        assertEquals(0.0, spectrum.trace(), 1e-9);
    }

    @Test void extractsRealEigenvaluesFromDiagonalSchurForm() {
        Matrix diagonal = Matrix.of(new double[][]{{-3, 0, 0}, {0, 2, 0}, {0, 0, 5}});
        SchurSpectrum spectrum = SchurSpectrum.of(RealSchurDecomposition.factor(diagonal));
        assertEquals(3, spectrum.eigenvalues().size());
        assertEquals(5.0, spectrum.spectralRadius(), 1e-9);
        assertEquals(5.0, spectrum.spectralAbscissa(), 1e-9);
        assertEquals(-30.0, spectrum.determinant(), 1e-9);
        assertEquals(4.0, spectrum.trace(), 1e-9);
    }

    @Test void complexEigenvalueHasConjugateAndMagnitudeOperations() {
        ComplexEigenvalue value = new ComplexEigenvalue(3, 4);
        assertEquals(5.0, value.magnitude(), 1e-12);
        assertEquals(-4.0, value.conjugate().imaginary(), 1e-12);
        assertFalse(value.isReal(1e-12));
        assertThrows(IllegalArgumentException.class, () -> new ComplexEigenvalue(Double.NaN, 0));
    }
}
