package tech.kayys.aljabr.linalg;

import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/** Cholesky factorization A = L*L^T for symmetric positive-definite matrices. */
public final class CholeskyDecomposition {
    private final Matrix lower;

    private CholeskyDecomposition(Matrix lower) { this.lower = lower; }

    public static CholeskyDecomposition factor(Matrix matrix) {
        return factor(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static CholeskyDecomposition factor(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        if (matrix.rows() != matrix.columns())
            throw new IllegalArgumentException("Cholesky factorization requires a square matrix");
        int n = matrix.rows();
        double[][] l = new double[n][n];
        double symmetryThreshold = tolerance.absolute() + tolerance.relative() * matrix.normInfinity();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (Math.abs(matrix.get(i, j) - matrix.get(j, i)) > symmetryThreshold)
                    throw new IllegalArgumentException("matrix must be symmetric within the configured tolerance");
                double sum = matrix.get(i, j);
                for (int k = 0; k < j; k++) sum = Math.fma(-l[i][k], l[j][k], sum);
                if (i == j) {
                    if (!Double.isFinite(sum) || sum <= symmetryThreshold)
                        throw new SingularMatrixException("matrix is not positive definite at diagonal " + i);
                    l[i][j] = Math.sqrt(sum);
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }
        return new CholeskyDecomposition(Matrix.of(l));
    }

    /** Returns the lower-triangular factor L. */
    public Matrix lower() { return lower; }

    public double determinant() {
        double product = 1.0;
        for (int i = 0; i < lower.rows(); i++) product *= lower.get(i, i);
        return product * product;
    }

    /** Solves A*x=b using forward and backward substitution. */
    public Vector solve(Vector rightHandSide) {
        Objects.requireNonNull(rightHandSide, "rightHandSide");
        int n = lower.rows();
        if (rightHandSide.size() != n)
            throw new IllegalArgumentException("right-hand side size must equal matrix size");
        double[] x = rightHandSide.toArray();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) x[i] = Math.fma(-lower.get(i, j), x[j], x[i]);
            x[i] /= lower.get(i, i);
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) x[i] = Math.fma(-lower.get(j, i), x[j], x[i]);
            x[i] /= lower.get(i, i);
        }
        return Vector.of(x);
    }
}
