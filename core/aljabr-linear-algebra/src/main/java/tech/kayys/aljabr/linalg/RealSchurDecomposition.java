package tech.kayys.aljabr.linalg;

import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Real Schur factorization A = Q*T*Q^T, where Q is orthogonal and T is
 * upper quasi-triangular (1x1 real blocks and 2x2 blocks for conjugate pairs).
 *
 * <p>This CPU reference implementation uses shifted QR iteration on a
 * Hessenberg reduction. Iteration is bounded; failure to converge is explicit.</p>
 */
public final class RealSchurDecomposition {
    private static final int MAX_ITERATIONS_PER_ORDER = 2000;

    private final Matrix t;
    private final Matrix q;

    private RealSchurDecomposition(Matrix t, Matrix q) {
        this.t = t;
        this.q = q;
    }

    public static RealSchurDecomposition factor(Matrix matrix) {
        return factor(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static RealSchurDecomposition factor(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        if (matrix.rows() != matrix.columns())
            throw new IllegalArgumentException("real Schur decomposition requires a square matrix");

        HessenbergReduction reduction = HessenbergReduction.reduce(matrix, tolerance);
        double[][] a = reduction.hessenberg().toArray();
        double[][] q = reduction.orthogonalQ().toArray();
        int n = a.length;
        double scale = Math.max(1.0, matrix.normInfinity());
        double threshold = tolerance.absolute() + tolerance.relative() * scale;
        int active = n;
        int iterations = 0;
        int limit = Math.max(1, MAX_ITERATIONS_PER_ORDER * n);

        while (active > 1) {
            double sub = Math.abs(a[active - 1][active - 2]);
            double local = Math.abs(a[active - 2][active - 2])
                    + Math.abs(a[active - 1][active - 1]);
            if (sub <= threshold + tolerance.relative() * local) {
                a[active - 1][active - 2] = 0.0;
                active--;
                continue;
            }

            if (active == 2 && hasComplexPair(a, 0, threshold)) break;

            // A converged trailing 2x2 block is a real Schur block when it
            // represents a complex-conjugate pair and is isolated below.
            if (active >= 3 && Math.abs(a[active - 2][active - 3]) <= threshold
                    && hasComplexPair(a, active - 2, threshold)) {
                a[active - 2][active - 3] = 0.0;
                active -= 2;
                continue;
            }

            if (++iterations > limit)
                throw new ArithmeticException("real Schur QR iteration did not converge within " + limit + " steps");

            double shift = wilkinsonShift(a, active, iterations, scale);
            double[][] block = leadingBlock(a, active);
            for (int i = 0; i < active; i++) block[i][i] -= shift;
            QR qr = qr(block);
            double[][] next = multiply(qr.r, qr.q);
            for (int i = 0; i < active; i++) next[i][i] += shift;

            // Apply the embedded similarity diag(Q_step, I)^T * A * diag(Q_step, I).
            // Updating the off-diagonal coupling is essential after deflation.
            double[][] leftUpdated = new double[active][n];
            for (int i = 0; i < active; i++)
                for (int column = 0; column < n; column++)
                    for (int k = 0; k < active; k++)
                        leftUpdated[i][column] = Math.fma(qr.q[k][i], a[k][column], leftUpdated[i][column]);
            for (int i = 0; i < active; i++)
                System.arraycopy(leftUpdated[i], 0, a[i], 0, n);
            for (int row = active; row < n; row++) {
                double[] updated = new double[active];
                for (int j = 0; j < active; j++)
                    for (int k = 0; k < active; k++)
                        updated[j] = Math.fma(a[row][k], qr.q[k][j], updated[j]);
                System.arraycopy(updated, 0, a[row], 0, active);
            }
            for (int i = 0; i < n; i++) {
                double[] row = new double[active];
                for (int j = 0; j < active; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < active; k++) sum = Math.fma(q[i][k], qr.q[k][j], sum);
                    row[j] = sum;
                }
                for (int j = 0; j < active; j++) q[i][j] = row[j];
            }

            // The exact QR similarity preserves Hessenberg structure. Remove
            // only round-off fill below the first subdiagonal.
            for (int i = 0; i < active; i++)
                for (int j = 0; j < i - 1; j++)
                    if (Math.abs(a[i][j]) <= 100.0 * threshold) a[i][j] = 0.0;
        }

        return new RealSchurDecomposition(Matrix.of(a), Matrix.of(q));
    }

    private static boolean hasComplexPair(double[][] a, int start, double tolerance) {
        double x = a[start][start], y = a[start][start + 1];
        double z = a[start + 1][start], w = a[start + 1][start + 1];
        double halfDifference = 0.5 * (x - w);
        double discriminant = halfDifference * halfDifference + y * z;
        double scale = Math.max(1.0, Math.max(Math.abs(x), Math.max(Math.abs(y),
                Math.max(Math.abs(z), Math.abs(w)))));
        return discriminant < -tolerance * scale;
    }

    private static double wilkinsonShift(double[][] a, int n, int iteration, double scale) {
        double x = a[n - 2][n - 2], y = a[n - 2][n - 1];
        double z = a[n - 1][n - 2], w = a[n - 1][n - 1];
        double half = 0.5 * (x - w);
        double discriminant = half * half + y * z;
        if (discriminant >= 0.0) {
            double root = Math.sqrt(discriminant);
            double first = 0.5 * (x + w) + root;
            double second = 0.5 * (x + w) - root;
            return Math.abs(first - w) < Math.abs(second - w) ? first : second;
        }
        // Real single-shift QR cannot split a complex pair by a real shift.
        // An alternating exceptional shift prevents a stationary trace/2 step.
        double perturbation = 0.01 * (1.0 + Math.abs(z)) * (iteration % 2 == 0 ? 1.0 : -1.0);
        return 0.5 * (x + w) + perturbation + Math.ulp(scale);
    }

    private static double[][] leadingBlock(double[][] a, int n) {
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(a[i], 0, result[i], 0, n);
        return result;
    }

    private static QR qr(double[][] source) {
        int n = source.length;
        double[][] r = leadingBlock(source, n);
        double[][] q = identity(n);
        for (int k = 0; k < n - 1; k++) {
            double norm = 0.0;
            for (int i = k; i < n; i++) norm = Math.hypot(norm, r[i][k]);
            if (norm == 0.0) continue;
            double alpha = -Math.copySign(norm, r[k][k]);
            double[] v = new double[n - k];
            for (int i = k; i < n; i++) v[i - k] = r[i][k];
            v[0] -= alpha;
            double vnorm = 0.0;
            for (double value : v) vnorm = Math.hypot(vnorm, value);
            if (vnorm == 0.0) continue;
            for (int i = 0; i < v.length; i++) v[i] /= vnorm;
            for (int column = k; column < n; column++) {
                double projection = 0.0;
                for (int i = 0; i < v.length; i++) projection = Math.fma(v[i], r[k + i][column], projection);
                for (int i = 0; i < v.length; i++) r[k + i][column] -= 2.0 * v[i] * projection;
            }
            for (int row = 0; row < n; row++) {
                double projection = 0.0;
                for (int i = 0; i < v.length; i++) projection = Math.fma(q[row][k + i], v[i], projection);
                for (int i = 0; i < v.length; i++) q[row][k + i] -= 2.0 * projection * v[i];
            }
        }
        return new QR(q, r);
    }

    private static double[][] identity(int n) {
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++) result[i][i] = 1.0;
        return result;
    }

    private static double[][] multiply(double[][] left, double[][] right) {
        int n = left.length;
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++)
                    result[i][j] = Math.fma(left[i][k], right[k][j], result[i][j]);
        return result;
    }

    public Matrix schurForm() { return t; }

    /** Orthogonal Q in A = Q*T*Q^T. */
    public Matrix orthogonalQ() { return q; }

    public Matrix reconstruct() { return q.multiply(t).multiply(q.transpose()); }

    /** True when the subdiagonal entry identifies a 2x2 real Schur block. */
    public boolean isComplexPairBlock(int index, Tolerance tolerance) {
        Objects.requireNonNull(tolerance, "tolerance");
        if (index < 0 || index + 1 >= t.rows()) throw new IndexOutOfBoundsException("block index");
        double sub = Math.abs(t.get(index + 1, index));
        double threshold = tolerance.absolute() + tolerance.relative() * Math.max(1.0, t.normInfinity());
        return sub > threshold && hasComplexPair(t.toArray(), index, threshold);
    }

    private record QR(double[][] q, double[][] r) {}
}
