package tech.kayys.aljabr.linalg;

import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Householder QR factorization for dense matrices with rows >= columns.
 * The factorization is A = Q*R, where Q is orthogonal and R is upper triangular.
 */
public final class QRDecomposition {
    private final double[][] qr;
    private final double[] tau;
    private final int rows;
    private final int columns;
    private final double threshold;

    private QRDecomposition(double[][] qr, double[] tau, int rows, int columns, double threshold) {
        this.qr = qr;
        this.tau = tau;
        this.rows = rows;
        this.columns = columns;
        this.threshold = threshold;
    }

    public static QRDecomposition factor(Matrix matrix) {
        return factor(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static QRDecomposition factor(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        int m = matrix.rows(), n = matrix.columns();
        if (m < n) throw new IllegalArgumentException("QR factorization requires rows >= columns");
        double[][] qr = matrix.toArray();
        double[] tau = new double[n];
        double threshold = tolerance.absolute() + tolerance.relative() * matrix.normInfinity();

        for (int k = 0; k < n; k++) {
            double norm = 0.0;
            for (int i = k; i < m; i++) norm = Math.hypot(norm, qr[i][k]);
            if (!Double.isFinite(norm) || norm <= threshold)
                throw new SingularMatrixException("matrix is rank-deficient at QR column " + k);

            double alpha = qr[k][k] >= 0.0 ? -norm : norm;
            double v0 = qr[k][k] - alpha;
            qr[k][k] = alpha;
            for (int i = k + 1; i < m; i++) qr[i][k] /= v0;
            tau[k] = (alpha - (alpha + v0)) / alpha;

            for (int j = k + 1; j < n; j++) {
                double dot = qr[k][j];
                for (int i = k + 1; i < m; i++) dot = Math.fma(qr[i][k], qr[i][j], dot);
                dot *= tau[k];
                qr[k][j] -= dot;
                for (int i = k + 1; i < m; i++) qr[i][j] = Math.fma(-qr[i][k], dot, qr[i][j]);
            }
        }
        return new QRDecomposition(qr, tau, m, n, threshold);
    }

    public Matrix r() {
        double[][] result = new double[columns][columns];
        for (int i = 0; i < columns; i++)
            for (int j = i; j < columns; j++) result[i][j] = qr[i][j];
        return Matrix.of(result);
    }

    /** Materializes the full m-by-m orthogonal Q matrix. */
    public Matrix q() {
        double[][] q = new double[rows][rows];
        for (int i = 0; i < rows; i++) q[i][i] = 1.0;
        for (int k = columns - 1; k >= 0; k--) {
            for (int j = 0; j < rows; j++) {
                double dot = q[k][j];
                for (int i = k + 1; i < rows; i++) dot = Math.fma(qr[i][k], q[i][j], dot);
                dot *= tau[k];
                q[k][j] -= dot;
                for (int i = k + 1; i < rows; i++) q[i][j] = Math.fma(-qr[i][k], dot, q[i][j]);
            }
        }
        return Matrix.of(q);
    }

    /** Solves the full-column-rank least-squares problem min ||A*x-b||_2. */
    public Vector solveLeastSquares(Vector rightHandSide) {
        Objects.requireNonNull(rightHandSide, "rightHandSide");
        if (rightHandSide.size() != rows)
            throw new IllegalArgumentException("right-hand side size must equal matrix row count");
        double[] y = rightHandSide.toArray();
        for (int k = 0; k < columns; k++) {
            double dot = y[k];
            for (int i = k + 1; i < rows; i++) dot = Math.fma(qr[i][k], y[i], dot);
            dot *= tau[k];
            y[k] -= dot;
            for (int i = k + 1; i < rows; i++) y[i] = Math.fma(-qr[i][k], dot, y[i]);
        }
        double[] x = new double[columns];
        for (int k = columns - 1; k >= 0; k--) {
            if (Math.abs(qr[k][k]) <= threshold)
                throw new SingularMatrixException("matrix is rank-deficient at QR diagonal " + k);
            double value = y[k];
            for (int j = k + 1; j < columns; j++) value = Math.fma(-qr[k][j], x[j], value);
            x[k] = value / qr[k][k];
        }
        return Vector.of(x);
    }
}
