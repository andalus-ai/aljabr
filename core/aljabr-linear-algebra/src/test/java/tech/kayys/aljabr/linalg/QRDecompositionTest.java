package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class QRDecompositionTest {
    @Test void reconstructsMatrixAndOrthogonalQ() {
        Matrix a = Matrix.of(new double[][]{{12, -51}, {6, 167}, {-4, 24}});
        QRDecomposition qr = QRDecomposition.factor(a);
        Matrix q = qr.q(), r = qr.r();
        Matrix reconstructed = q.multiply(r);
        for (int i = 0; i < a.rows(); i++)
            for (int j = 0; j < a.columns(); j++)
                assertEquals(a.get(i, j), reconstructed.get(i, j), 1e-10);
        assertEquals(a.rows(), q.rows());
        assertEquals(a.columns(), q.columns());
        Matrix identity = q.transpose().multiply(q);
        for (int i = 0; i < identity.rows(); i++)
            for (int j = 0; j < identity.columns(); j++)
                assertEquals(i == j ? 1.0 : 0.0, identity.get(i, j), 1e-10);
    }

    @Test void solvesOverdeterminedLeastSquares() {
        Matrix a = Matrix.of(new double[][]{{1, 1}, {1, 2}, {1, 3}});
        Vector b = Vector.of(1, 2, 2);
        Vector x = QRDecomposition.factor(a).solveLeastSquares(b);
        assertEquals(2.0 / 3.0, x.get(0), 1e-10);
        assertEquals(0.5, x.get(1), 1e-10);
    }

    @Test void rejectsWideAndRankDeficientMatrices() {
        assertThrows(IllegalArgumentException.class,
                () -> QRDecomposition.factor(Matrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}})));
        assertThrows(SingularMatrixException.class,
                () -> QRDecomposition.factor(Matrix.of(new double[][]{{1, 2}, {2, 4}, {3, 6}})));
    }
}
