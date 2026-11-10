package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.numerics.Tolerance;
import static org.junit.jupiter.api.Assertions.*;

class LUDecompositionTest {
    @Test void solvesSystemWithPivoting() {
        Matrix a = Matrix.of(new double[][]{{0, 2}, {1, 1}});
        Vector b = Vector.of(4, 3);
        Vector x = LUDecomposition.factor(a).solve(b);
        assertEquals(1.0, x.get(0), 1e-12);
        assertEquals(2.0, x.get(1), 1e-12);
        assertEquals(-2.0, LUDecomposition.factor(a).determinant(), 1e-12);
    }

    @Test void solvesMultipleRightHandSides() {
        Matrix a = Matrix.of(new double[][]{{3, 1}, {1, 2}});
        Matrix b = Matrix.of(new double[][]{{9, 1}, {8, 0}});
        Matrix x = LUDecomposition.factor(a).solve(b);
        Matrix reconstructed = a.multiply(x);
        assertEquals(9.0, reconstructed.get(0, 0), 1e-12);
        assertEquals(8.0, reconstructed.get(1, 0), 1e-12);
        assertEquals(1.0, reconstructed.get(0, 1), 1e-12);
        assertEquals(0.0, reconstructed.get(1, 1), 1e-12);
    }

    @Test void rejectsSingularAndNonsquareMatrices() {
        assertThrows(SingularMatrixException.class,
                () -> LUDecomposition.factor(Matrix.of(new double[][]{{1, 2}, {2, 4}})));
        assertThrows(IllegalArgumentException.class,
                () -> LUDecomposition.factor(Matrix.of(new double[][]{{1, 2, 3}})));
    }

    @Test void allowsExplicitTolerancePolicy() {
        assertThrows(SingularMatrixException.class,
                () -> LUDecomposition.factor(Matrix.of(new double[][]{{1, 0}, {0, 1e-9}}), new Tolerance(1e-8, 0)));
    }
}
