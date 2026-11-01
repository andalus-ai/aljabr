package tech.kayys.aljabr.linalg;

import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Symmetric eigendecomposition using cyclic Jacobi rotations.
 * Eigenvalues are sorted in descending order and eigenvectors are columns.
 */
public final class SymmetricEigenDecomposition {
    private static final int MAX_SWEEPS = 100;

    private final double[] eigenvalues;
    private final Matrix eigenvectors;

    private SymmetricEigenDecomposition(double[] eigenvalues, Matrix eigenvectors) {
        this.eigenvalues = eigenvalues;
        this.eigenvectors = eigenvectors;
    }

    public static SymmetricEigenDecomposition factor(Matrix matrix) {
        return factor(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static SymmetricEigenDecomposition factor(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        if (matrix.rows() != matrix.columns())
            throw new IllegalArgumentException("symmetric eigendecomposition requires a square matrix");

        int n = matrix.rows();
        double scale = matrix.normInfinity();
        double threshold = tolerance.absolute() + tolerance.relative() * scale;
        double[][] a = matrix.toArray();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(a[i][j] - a[j][i]) > threshold)
                    throw new IllegalArgumentException("matrix must be symmetric within the configured tolerance");
                double average = 0.5 * a[i][j] + 0.5 * a[j][i];
                a[i][j] = average;
                a[j][i] = average;
            }
        }

        double[][] vectors = new double[n][n];
        for (int i = 0; i < n; i++) vectors[i][i] = 1.0;

        boolean converged = n < 2;
        for (int sweep = 0; sweep < MAX_SWEEPS && !converged; sweep++) {
            double maximum = 0.0;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = a[p][q];
                    maximum = Math.max(maximum, Math.abs(apq));
                    if (Math.abs(apq) <= threshold) continue;

                    double tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                    double t = Math.copySign(1.0, tau) /
                            (Math.abs(tau) + Math.hypot(1.0, tau));
                    double c = 1.0 / Math.hypot(1.0, t);
                    double s = t * c;
                    double app = a[p][p], aqq = a[q][q];
                    a[p][p] = app - t * apq;
                    a[q][q] = aqq + t * apq;
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;

                    for (int k = 0; k < n; k++) {
                        if (k != p && k != q) {
                            double akp = a[k][p], akq = a[k][q];
                            double newP = c * akp - s * akq;
                            double newQ = s * akp + c * akq;
                            a[k][p] = newP; a[p][k] = newP;
                            a[k][q] = newQ; a[q][k] = newQ;
                        }
                        double vkp = vectors[k][p], vkq = vectors[k][q];
                        vectors[k][p] = c * vkp - s * vkq;
                        vectors[k][q] = s * vkp + c * vkq;
                    }
                }
            }
            if (maximum <= threshold) converged = true;
        }
        if (!converged) throw new ArithmeticException(
                "Jacobi symmetric eigendecomposition did not converge within " + MAX_SWEEPS + " sweeps");

        // A final off-diagonal check avoids accepting a sweep that only became small after rotations.
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (Math.abs(a[i][j]) > threshold)
                    throw new ArithmeticException("Jacobi symmetric eigendecomposition did not converge");

        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = 0; i < n - 1; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++)
                if (a[order[j]][order[j]] > a[order[best]][order[best]]) best = j;
            int swap = order[i]; order[i] = order[best]; order[best] = swap;
        }

        double[] values = new double[n];
        double[][] sortedVectors = new double[n][n];
        for (int j = 0; j < n; j++) {
            int source = order[j];
            values[j] = a[source][source];
            for (int i = 0; i < n; i++) sortedVectors[i][j] = vectors[i][source];
        }
        return new SymmetricEigenDecomposition(values, Matrix.of(sortedVectors));
    }

    public double[] eigenvalues() { return eigenvalues.clone(); }

    /** Orthonormal eigenvectors stored as columns, ordered with eigenvalues(). */
    public Matrix eigenvectors() { return eigenvectors; }

    /** Reconstructs V * diag(lambda) * V^T. */
    public Matrix reconstruct() {
        int n = eigenvalues.length;
        double[][] scaled = eigenvectors.toArray();
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) scaled[i][j] *= eigenvalues[j];
        return Matrix.of(scaled).multiply(eigenvectors.transpose());
    }
}
