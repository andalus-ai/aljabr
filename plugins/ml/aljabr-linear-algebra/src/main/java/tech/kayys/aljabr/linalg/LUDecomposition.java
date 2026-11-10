package tech.kayys.aljabr.linalg;

import java.util.Objects;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import tech.kayys.aljabr.numerics.Tolerance;

/**
 * Dense LU factorization with partial pivoting (P*A = L*U).
 * This is a straightforward double-precision CPU reference implementation.
 */
public final class LUDecomposition {
    private final double[][] lu;
    private final int[] pivot;
    private final int pivotSign;

    private LUDecomposition(double[][] lu, int[] pivot, int pivotSign) {
        this.lu = lu;
        this.pivot = pivot;
        this.pivotSign = pivotSign;
    }

    public static LUDecomposition factor(Matrix matrix) {
        return factor(matrix, NumericalPrecision.FLOAT64.defaultTolerance());
    }

    public static LUDecomposition factor(Matrix matrix, Tolerance tolerance) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(tolerance, "tolerance");
        if (matrix.rows() != matrix.columns()) throw new IllegalArgumentException("LU factorization requires a square matrix");
        int n = matrix.rows();
        double[][] lu = matrix.toArray();
        int[] pivot = new int[n];
        for (int i = 0; i < n; i++) pivot[i] = i;
        double threshold = tolerance.absolute() + tolerance.relative() * matrix.normInfinity();
        int sign = 1;

        for (int column = 0; column < n; column++) {
            int best = column;
            double largest = Math.abs(lu[column][column]);
            for (int row = column + 1; row < n; row++) {
                double candidate = Math.abs(lu[row][column]);
                if (candidate > largest) { largest = candidate; best = row; }
            }
            if (!Double.isFinite(largest) || largest <= threshold)
                throw new SingularMatrixException("matrix is singular or numerically rank-deficient at column " + column);

            if (best != column) {
                double[] row = lu[best]; lu[best] = lu[column]; lu[column] = row;
                int p = pivot[best]; pivot[best] = pivot[column]; pivot[column] = p;
                sign = -sign;
            }

            for (int row = column + 1; row < n; row++) {
                lu[row][column] /= lu[column][column];
                for (int k = column + 1; k < n; k++)
                    lu[row][k] = Math.fma(-lu[row][column], lu[column][k], lu[row][k]);
            }
        }
        return new LUDecomposition(lu, pivot, sign);
    }

    public int size() { return pivot.length; }

    public double determinant() {
        double determinant = pivotSign;
        for (int i = 0; i < size(); i++) determinant *= lu[i][i];
        return determinant;
    }

    /** Solves A*x=b using the stored pivoted factorization. */
    public Vector solve(Vector rightHandSide) {
        Objects.requireNonNull(rightHandSide, "rightHandSide");
        if (rightHandSide.size() != size()) throw new IllegalArgumentException("right-hand side size must equal matrix size");
        double[] x = new double[size()];
        for (int i = 0; i < size(); i++) x[i] = rightHandSide.get(pivot[i]);

        for (int column = 0; column < size(); column++)
            for (int row = column + 1; row < size(); row++)
                x[row] = Math.fma(-lu[row][column], x[column], x[row]);

        for (int column = size() - 1; column >= 0; column--) {
            x[column] /= lu[column][column];
            for (int row = 0; row < column; row++)
                x[row] = Math.fma(-lu[row][column], x[column], x[row]);
        }
        return Vector.of(x);
    }

    /** Solves A*X=B for all columns of B using one factorization. */
    public Matrix solve(Matrix rightHandSides) {
        Objects.requireNonNull(rightHandSides, "rightHandSides");
        if (rightHandSides.rows() != size()) throw new IllegalArgumentException("right-hand side row count must equal matrix size");
        double[][] result = new double[size()][rightHandSides.columns()];
        for (int column = 0; column < rightHandSides.columns(); column++) {
            double[] values = new double[size()];
            for (int row = 0; row < size(); row++) values[row] = rightHandSides.get(row, column);
            Vector solution = solve(Vector.of(values));
            for (int row = 0; row < size(); row++) result[row][column] = solution.get(row);
        }
        return Matrix.of(result);
    }
}
