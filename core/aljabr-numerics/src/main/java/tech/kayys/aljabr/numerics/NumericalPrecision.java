package tech.kayys.aljabr.numerics;

/** Common floating-point formats and practical comparison tolerances. */
public enum NumericalPrecision {
    FLOAT32(new Tolerance(1.0e-6, 1.0e-5)),
    FLOAT64(new Tolerance(1.0e-12, 1.0e-10));

    private final Tolerance defaultTolerance;

    NumericalPrecision(Tolerance defaultTolerance) {
        this.defaultTolerance = defaultTolerance;
    }

    public Tolerance defaultTolerance() {
        return defaultTolerance;
    }

    public int bits() {
        return this == FLOAT32 ? Float.SIZE : Double.SIZE;
    }
}
