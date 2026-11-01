package tech.kayys.aljabr.linalg;

import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Reduces a real square matrix to upper Hessenberg form by orthogonal
 * Householder similarity transformations: H = Q^T * A * Q.
 *
 * <p>For symmetric input the result is tridiagonal up to round-off. This
 * transformation is the standard preprocessing stage for general real
 * Schur/eigenvalue algorithms.</p>
 */
public final class HessenbergReduction {
    private final Matrix hessenberg;
    private final Matrix orthogonalQ;

    private HessenbergReduction(Matrix hessenberg, Matrix orthogonalQ) {
        this.hessenberg = hessenberg;
        this.orthogonalQ = orthogonalQ;
    }

    public static HessenbergReduction reduce(Matrix matrix) {
        return reduce(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static HessenbergReduction reduce(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        if (matrix.rows() != matrix.columns())
            throw new IllegalArgumentException("Hessenberg reduction requires a square matrix");

        int n = matrix.rows();
        double[][] a = matrix.toArray();
        double[][] q = Matrix.identity(n).toArray();
        double threshold = tolerance.absolute() + tolerance.relative() * matrix.normInfinity();

        for (int k = 0; k < n - 2; k++) {
            int start = k + 1;
            double norm = 0.0;
            for (int i = start; i < n; i++) norm = Math.hypot(norm, a[i][k]);
            if (norm <= threshold) continue;

            double alpha = -Math.copySign(norm, a[start][k]);
            double[] v = new double[n - start];
            for (int i = start; i < n; i++) v[i - start] = a[i][k];
            v[0] -= alpha;

            double vNorm = 0.0;
            for (double value : v) vNorm = Math.hypot(vNorm, value);
            if (vNorm == 0.0) continue;
            for (int i = 0; i < v.length; i++) v[i] /= vNorm;

            // Left application: A <- P*A, where P = I - 2*v*v^T.
            for (int column = 0; column < n; column++) {
                double projection = 0.0;
                for (int i = 0; i < v.length; i++)
                    projection = Math.fma(v[i], a[start + i][column], projection);
                for (int i = 0; i < v.length; i++)
                    a[start + i][column] = Math.fma(-2.0 * v[i], projection, a[start + i][column]);
            }

            // Right application: A <- A*P.
            for (int row = 0; row < n; row++) {
                double projection = 0.0;
                for (int i = 0; i < v.length; i++)
                    projection = Math.fma(a[row][start + i], v[i], projection);
                for (int i = 0; i < v.length; i++)
                    a[row][start + i] = Math.fma(-2.0 * projection, v[i], a[row][start + i]);
            }

            // Accumulate Q <- Q*P.
            for (int row = 0; row < n; row++) {
                double projection = 0.0;
                for (int i = 0; i < v.length; i++)
                    projection = Math.fma(q[row][start + i], v[i], projection);
                for (int i = 0; i < v.length; i++)
                    q[row][start + i] = Math.fma(-2.0 * projection, v[i], q[row][start + i]);
            }

            // Enforce the structural zeros introduced by the reflector.
            for (int row = start + 1; row < n; row++) a[row][k] = 0.0;
            a[start][k] = alpha;
        }

        return new HessenbergReduction(Matrix.of(a), Matrix.of(q));
    }

    /** Upper Hessenberg matrix; entries below the first subdiagonal are zero. */
    public Matrix hessenberg() { return hessenberg; }

    /** Orthogonal accumulated transformation Q, satisfying H = Q^T*A*Q. */
    public Matrix orthogonalQ() { return orthogonalQ; }

    /** Reconstructs the original matrix as Q*H*Q^T. */
    public Matrix reconstruct() {
        return orthogonalQ.multiply(hessenberg).multiply(orthogonalQ.transpose());
    }
}
