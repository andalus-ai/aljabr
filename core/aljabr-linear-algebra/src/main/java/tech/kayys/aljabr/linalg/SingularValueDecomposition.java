package tech.kayys.aljabr.linalg;

import java.util.Arrays;
import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Thin singular value decomposition using a one-sided Jacobi CPU algorithm.
 * Returns A = U * diag(s) * V^T with descending, non-negative singular values.
 */
public final class SingularValueDecomposition {
    private static final int MAX_SWEEPS = 100;

    private final Matrix u;
    private final Matrix v;
    private final double[] singularValues;
    private final double rankThreshold;

    private SingularValueDecomposition(Matrix u, double[] singularValues, Matrix v, double rankThreshold) {
        this.u = u;
        this.v = v;
        this.singularValues = singularValues;
        this.rankThreshold = rankThreshold;
    }

    public static SingularValueDecomposition factor(Matrix matrix) {
        return factor(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static SingularValueDecomposition factor(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        int m = matrix.rows(), n = matrix.columns();
        if (m >= n) return factorTall(matrix, tolerance);

        SingularValueDecomposition transposed = factorTall(matrix.transpose(), tolerance);
        return new SingularValueDecomposition(
                transposed.v, transposed.singularValues, transposed.u, transposed.rankThreshold);
    }

    private static SingularValueDecomposition factorTall(Matrix matrix, Tolerance tolerance) {
        int m = matrix.rows(), n = matrix.columns();
        double[][] b = matrix.toArray();
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;

        double correlationTolerance = Math.max(1.0e-15, tolerance.relative());
        boolean converged = n < 2;
        for (int sweep = 0; sweep < MAX_SWEEPS && !converged; sweep++) {
            boolean rotated = false;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double alpha = 0.0, beta = 0.0, gamma = 0.0;
                    for (int i = 0; i < m; i++) {
                        alpha = Math.fma(b[i][p], b[i][p], alpha);
                        beta = Math.fma(b[i][q], b[i][q], beta);
                        gamma = Math.fma(b[i][p], b[i][q], gamma);
                    }
                    if (alpha == 0.0 || beta == 0.0
                            || Math.abs(gamma) <= correlationTolerance * Math.sqrt(alpha) * Math.sqrt(beta)) {
                        continue;
                    }

                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double t = Math.copySign(1.0, zeta) /
                            (Math.abs(zeta) + Math.hypot(1.0, zeta));
                    double c = 1.0 / Math.hypot(1.0, t);
                    double s = c * t;
                    rotateColumns(b, p, q, c, s);
                    rotateColumns(v, p, q, c, s);
                    rotated = true;
                }
            }
            converged = !rotated;
        }
        if (!converged) throw new ArithmeticException("Jacobi SVD did not converge within " + MAX_SWEEPS + " sweeps");

        double[] norms = new double[n];
        for (int j = 0; j < n; j++) {
            double norm = 0.0;
            for (int i = 0; i < m; i++) norm = Math.hypot(norm, b[i][j]);
            norms[j] = norm;
        }

        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (left, right) -> Double.compare(norms[right], norms[left]));

        double[] singularValues = new double[n];
        double[][] u = new double[m][n];
        double[][] sortedV = new double[n][n];
        for (int j = 0; j < n; j++) {
            int source = order[j];
            double sigma = norms[source];
            singularValues[j] = sigma;
            for (int i = 0; i < n; i++) sortedV[i][j] = v[i][source];
            if (sigma > 0.0) {
                for (int i = 0; i < m; i++) u[i][j] = b[i][source] / sigma;
            }
        }
        double threshold = tolerance.absolute() + tolerance.relative()
                * (singularValues.length == 0 ? 0.0 : singularValues[0]);
        return new SingularValueDecomposition(Matrix.of(u), singularValues, Matrix.of(sortedV), threshold);
    }

    private static void rotateColumns(double[][] values, int p, int q, double c, double s) {
        for (double[] row : values) {
            double x = row[p], y = row[q];
            row[p] = c * x - s * y;
            row[q] = s * x + c * y;
        }
    }

    /** Thin left singular vectors, with shape m-by-min(m,n). */
    public Matrix u() { return u; }

    /** Thin right singular vectors, with shape n-by-min(m,n). */
    public Matrix v() { return v; }

    public double[] singularValues() { return singularValues.clone(); }

    /** Numerical rank under the tolerance supplied to factor(). */
    public int rank() {
        int rank = 0;
        for (double value : singularValues) if (value > rankThreshold) rank++;
        return rank;
    }

    /** Moore-Penrose least-squares solution using the tolerance-truncated pseudoinverse. */
    public Vector solve(Vector rightHandSide) {
        Objects.requireNonNull(rightHandSide, "rightHandSide");
        if (rightHandSide.size() != u.rows())
            throw new IllegalArgumentException("right-hand side size must equal original matrix row count");
        double[] result = new double[v.rows()];
        for (int j = 0; j < singularValues.length; j++) {
            double sigma = singularValues[j];
            if (sigma <= rankThreshold) continue;
            double projection = 0.0;
            for (int i = 0; i < u.rows(); i++)
                projection = Math.fma(u.get(i, j), rightHandSide.get(i), projection);
            double coefficient = projection / sigma;
            for (int i = 0; i < v.rows(); i++)
                result[i] = Math.fma(v.get(i, j), coefficient, result[i]);
        }
        return Vector.of(result);
    }
}
