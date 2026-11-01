package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CholeskyDecompositionTest {
    @Test void factorsAndSolvesPositiveDefiniteMatrix() {
        Matrix a = Matrix.of(new double[][]{{4, 2}, {2, 3}});
        CholeskyDecomposition cholesky = CholeskyDecomposition.factor(a);
        Matrix l = cholesky.lower();
        Matrix reconstructed = l.multiply(l.transpose());
        for (int i = 0; i < a.rows(); i++)
            for (int j = 0; j < a.columns(); j++)
                assertEquals(a.get(i, j), reconstructed.get(i, j), 1e-12);
        Vector x = cholesky.solve(Vector.of(6, 5));
        assertEquals(1.0, x.get(0), 1e-12);
        assertEquals(1.0, x.get(1), 1e-12);
        assertEquals(8.0, cholesky.determinant(), 1e-12);
    }

    @Test void rejectsNonsymmetricAndNonPositiveDefiniteMatrices() {
        assertThrows(IllegalArgumentException.class,
                () -> CholeskyDecomposition.factor(Matrix.of(new double[][]{{1, 2}, {0, 1}})));
        assertThrows(SingularMatrixException.class,
                () -> CholeskyDecomposition.factor(Matrix.of(new double[][]{{1, 2}, {2, 1}})));
    }
}
