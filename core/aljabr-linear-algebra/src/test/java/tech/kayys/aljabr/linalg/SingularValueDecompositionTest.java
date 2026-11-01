package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SingularValueDecompositionTest {
    @Test void reconstructsTallMatrixAndReturnsOrthonormalVectors() {
        Matrix a = Matrix.of(new double[][]{{3, 1}, {1, 3}, {1, 1}});
        SingularValueDecomposition svd = SingularValueDecomposition.factor(a);
        double[] s = svd.singularValues();
        assertTrue(s[0] >= s[1]);
        Matrix reconstructed = svd.u().multiply(Matrix.of(new double[][]{{s[0], 0}, {0, s[1]}}))
                .multiply(svd.v().transpose());
        assertMatrixClose(a, reconstructed, 1e-9);
        assertMatrixClose(Matrix.identity(2), svd.u().transpose().multiply(svd.u()), 1e-9);
        assertMatrixClose(Matrix.identity(2), svd.v().transpose().multiply(svd.v()), 1e-9);
    }

    @Test void supportsWideMatrices() {
        Matrix a = Matrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        SingularValueDecomposition svd = SingularValueDecomposition.factor(a);
        double[] s = svd.singularValues();
        Matrix diagonal = Matrix.of(new double[][]{{s[0], 0}, {0, s[1]}, {0, 0}});
        Matrix reconstructed = svd.u().multiply(diagonal).multiply(svd.v().transpose());
        assertMatrixClose(a, reconstructed, 1e-9);
        assertEquals(2, svd.u().rows());
        assertEquals(2, svd.u().columns());
        assertEquals(3, svd.v().rows());
        assertEquals(2, svd.v().columns());
    }

    @Test void computesRankAndMinimumNormSolutionForRankDeficientMatrix() {
        Matrix a = Matrix.of(new double[][]{{1, 2}, {2, 4}, {3, 6}});
        SingularValueDecomposition svd = SingularValueDecomposition.factor(a);
        assertEquals(1, svd.rank());
        Vector x = svd.solve(Vector.of(1, 2, 3));
        Vector residual = a.multiply(x).subtract(Vector.of(1, 2, 3));
        assertTrue(residual.norm2() < 1e-9);
        assertEquals(0.2, x.get(0), 1e-8);
        assertEquals(0.4, x.get(1), 1e-8);
    }

    private static void assertMatrixClose(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.rows(), actual.rows());
        assertEquals(expected.columns(), actual.columns());
        for (int i = 0; i < expected.rows(); i++)
            for (int j = 0; j < expected.columns(); j++)
                assertEquals(expected.get(i, j), actual.get(i, j), tolerance,
                        "mismatch at (" + i + ", " + j + ")");
    }
}
