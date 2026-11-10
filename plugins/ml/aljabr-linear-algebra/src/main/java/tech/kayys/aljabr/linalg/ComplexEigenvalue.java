package tech.kayys.aljabr.linalg;

/** A real or complex eigenvalue represented by its Cartesian components. */
public record ComplexEigenvalue(double real, double imaginary) {
    public ComplexEigenvalue {
        if (!Double.isFinite(real) || !Double.isFinite(imaginary))
            throw new IllegalArgumentException("eigenvalue components must be finite");
    }

    public double magnitude() { return Math.hypot(real, imaginary); }

    public boolean isReal(double tolerance) {
        if (!Double.isFinite(tolerance) || tolerance < 0.0)
            throw new IllegalArgumentException("tolerance must be finite and non-negative");
        return Math.abs(imaginary) <= tolerance;
    }

    public ComplexEigenvalue conjugate() {
        return new ComplexEigenvalue(real, -imaginary);
    }
}
