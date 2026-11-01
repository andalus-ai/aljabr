package tech.kayys.aljabr.numerics;

/**
 * Combined absolute and relative tolerance for floating-point comparisons.
 *
 * <p>Two finite values are close when their absolute difference is no greater
 * than max(absoluteTolerance, relativeTolerance * max(abs(a), abs(b))).</p>
 */
public record Tolerance(double absolute, double relative) {
    public Tolerance {
        if (!Double.isFinite(absolute) || absolute < 0.0) {
            throw new IllegalArgumentException("absolute tolerance must be finite and non-negative");
        }
        if (!Double.isFinite(relative) || relative < 0.0) {
            throw new IllegalArgumentException("relative tolerance must be finite and non-negative");
        }
    }

    public boolean isClose(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
        }
        double difference = Math.abs(a - b);
        double scale = Math.max(Math.abs(a), Math.abs(b));
        return difference <= Math.max(absolute, relative * scale);
    }
}
