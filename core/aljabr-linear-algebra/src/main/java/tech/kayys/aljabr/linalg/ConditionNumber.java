package tech.kayys.aljabr.linalg;

import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Condition diagnostics derived from the singular values of a dense matrix.
 *
 * <p>The 2-norm condition number is sigma_max / sigma_min. A rectangular
 * matrix is treated as full rank only when all min(rows, columns) singular
 * values exceed the supplied SVD rank threshold. Rank-deficient matrices,
 * including the zero matrix, have infinite condition number.</p>
 *
 * <p>This is a dense CPU reference implementation. It is intended for
 * diagnostics and algorithm selection, not as a replacement for a
 * LAPACK condition estimator.</p>
 */
public final class ConditionNumber {
    private final double twoNorm;
    private final double reciprocalTwoNorm;
    private final int numericalRank;
    private final int maximumRank;

    private ConditionNumber(double twoNorm, int numericalRank, int maximumRank) {
        this.twoNorm = twoNorm;
        this.reciprocalTwoNorm = Double.isFinite(twoNorm) && twoNorm > 0.0 ? 1.0 / twoNorm : 0.0;
        this.numericalRank = numericalRank;
        this.maximumRank = maximumRank;
    }

    public static ConditionNumber of(Matrix matrix) {
        return of(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static ConditionNumber of(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        SingularValueDecomposition svd = SingularValueDecomposition.factor(matrix, tolerance);
        double[] singularValues = svd.singularValues();
        int maximumRank = Math.min(matrix.rows(), matrix.columns());
        double condition = Double.POSITIVE_INFINITY;
        if (maximumRank > 0 && svd.rank() == maximumRank
                && singularValues[maximumRank - 1] > 0.0) {
            condition = singularValues[0] / singularValues[maximumRank - 1];
        }
        return new ConditionNumber(condition, svd.rank(), maximumRank);
    }

    /** Spectral (2-norm) condition number; positive infinity means numerically rank deficient. */
    public double twoNorm() { return twoNorm; }

    /** Reciprocal spectral condition number; zero means numerically rank deficient. */
    public double reciprocalTwoNorm() { return reciprocalTwoNorm; }

    public int numericalRank() { return numericalRank; }

    public int maximumRank() { return maximumRank; }

    public boolean isNumericallyRankDeficient() { return numericalRank < maximumRank; }
}
