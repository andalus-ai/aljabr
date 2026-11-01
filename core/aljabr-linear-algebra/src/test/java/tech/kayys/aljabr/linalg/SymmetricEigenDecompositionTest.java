package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SymmetricEigenDecompositionTest {
    @Test void reconstructsSymmetricMatrixAndReturnsOrthonormalEigenvectors() {
        Matrix a = Matrix.of(new double[][]{{4, 1, 1}, {1, 3, 0}, {1, 0, 2}});
        SymmetricEigenDecomposition eigen = SymmetricEigenDecomposition.factor(a);
        double[] values = eigen.eigenvalues();
        for (int i = 1; i < values.length; i++) assertTrue(values[i - 1] >= values[i]);
        assertMatrixClose(a, eigen.reconstruct(), 1e-9);
        assertMatrixClose(Matrix.identity(3),
                eigen.eigenvectors().transpose().multiply(eigen.eigenvectors()), 1e-9);
    }

    @Test void handlesDiagonalAndRepeatedEigenvalues() {
        Matrix a = Matrix.of(new double[][]{{5, 0, 0}, {0, 2, 0}, {0, 0, 2}});
        SymmetricEigenDecomposition eigen = SymmetricEigenDecomposition.factor(a);
        assertArrayEquals(new double[]{5, 2, 2}, eigen.eigenvalues(), 1e-12);
        assertMatrixClose(a, eigen.reconstruct(), 1e-12);
    }

    @Test void rejectsNonsquareAndNonsymmetricMatrices() {
        assertThrows(IllegalArgumentException.class,
                () -> SymmetricEigenDecomposition.factor(Matrix.of(new double[][]{{1, 2, 3}, {2, 3, 4}})));
        assertThrows(IllegalArgumentException.class,
                () -> SymmetricEigenDecomposition.factor(Matrix.of(new double[][]{{1, 2}, {0, 1}})));
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
