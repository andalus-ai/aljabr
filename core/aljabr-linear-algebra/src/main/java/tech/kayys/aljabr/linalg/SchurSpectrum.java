package tech.kayys.aljabr.linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Eigenvalue-level diagnostics extracted from a real Schur form.
 *
 * <p>Complex-conjugate pairs are represented explicitly. Eigenvalues are
 * returned in Schur block order; no magnitude or algebraic sorting is implied.</p>
 */
public final class SchurSpectrum {
    private final List<ComplexEigenvalue> eigenvalues;
    private final double spectralRadius;
    private final double spectralAbscissa;
    private final double trace;
    private final double determinant;

    private SchurSpectrum(List<ComplexEigenvalue> eigenvalues, double spectralRadius,
                          double spectralAbscissa, double trace, double determinant) {
        this.eigenvalues = List.copyOf(eigenvalues);
        this.spectralRadius = spectralRadius;
        this.spectralAbscissa = spectralAbscissa;
        this.trace = trace;
        this.determinant = determinant;
    }

    public static SchurSpectrum of(RealSchurDecomposition decomposition) {
        return of(decomposition, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static SchurSpectrum of(RealSchurDecomposition decomposition, Tolerance tolerance) {
        Objects.requireNonNull(decomposition, "decomposition");
        Objects.requireNonNull(tolerance, "tolerance");
        Matrix t = decomposition.schurForm();
        int n = t.rows();
        if (n != t.columns()) throw new IllegalArgumentException("Schur form must be square");

        double threshold = tolerance.absolute()
                + tolerance.relative() * Math.max(1.0, t.normInfinity());
        List<ComplexEigenvalue> values = new ArrayList<>(n);
        double radius = 0.0;
        double abscissa = Double.NEGATIVE_INFINITY;
        double trace = 0.0;
        double determinant = 1.0;

        for (int i = 0; i < n;) {
            if (i + 1 < n && Math.abs(t.get(i + 1, i)) > threshold) {
                double a = t.get(i, i);
                double b = t.get(i, i + 1);
                double c = t.get(i + 1, i);
                double d = t.get(i + 1, i + 1);
                double halfDifference = 0.5 * (a - d);
                double halfTrace = 0.5 * (a + d);
                double discriminant = halfDifference * halfDifference + b * c;
                if (discriminant < 0.0) {
                    double imaginary = Math.sqrt(-discriminant);
                    values.add(new ComplexEigenvalue(halfTrace, imaginary));
                    values.add(new ComplexEigenvalue(halfTrace, -imaginary));
                } else {
                    double root = Math.sqrt(discriminant);
                    values.add(new ComplexEigenvalue(halfTrace + root, 0.0));
                    values.add(new ComplexEigenvalue(halfTrace - root, 0.0));
                }
                trace += a + d;
                determinant *= Math.fma(a, d, -b * c);
                i += 2;
            } else {
                double value = t.get(i, i);
                values.add(new ComplexEigenvalue(value, 0.0));
                trace += value;
                determinant *= value;
                i++;
            }
        }

        for (ComplexEigenvalue value : values) {
            radius = Math.max(radius, value.magnitude());
            abscissa = Math.max(abscissa, value.real());
        }
        return new SchurSpectrum(values, radius, abscissa, trace, determinant);
    }

    /** Eigenvalues in real-Schur block order; complex pairs are adjacent. */
    public List<ComplexEigenvalue> eigenvalues() { return eigenvalues; }

    /** Maximum absolute eigenvalue, useful for discrete-time stability checks. */
    public double spectralRadius() { return spectralRadius; }

    /** Largest real part of any eigenvalue, useful for continuous-time stability checks. */
    public double spectralAbscissa() { return spectralAbscissa; }

    /** Sum of eigenvalues, equal to the matrix trace up to rounding. */
    public double trace() { return trace; }

    /** Product of eigenvalues, equal to the matrix determinant up to rounding. */
    public double determinant() { return determinant; }
}
