package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.numerics.NumericalPrecision;
import static org.junit.jupiter.api.Assertions.*;

class RealSchurDecompositionTest {
    @Test void decomposesRealSpectrumAndReconstructsInput() {
        Matrix a = Matrix.of(new double[][]{{4, 1, 2}, {1, 3, 0}, {2, 0, 2}});
        RealSchurDecomposition result = RealSchurDecomposition.factor(a);
        assertClose(a, result.reconstruct(), 1e-7);
        assertClose(Matrix.identity(3), result.orthogonalQ().transpose().multiply(result.orthogonalQ()), 1e-7);
        assertQuasiTriangular(result.schurForm(), 1e-7);
    }

    @Test void preservesTwoByTwoComplexConjugateBlock() {
        Matrix a = Matrix.of(new double[][]{{0, -1}, {1, 0}});
        RealSchurDecomposition result = RealSchurDecomposition.factor(a);
        assertClose(a, result.reconstruct(), 1e-9);
        assertTrue(result.isComplexPairBlock(0, NumericalPrecision.FLOAT64.defaultTolerance()));
        assertQuasiTriangular(result.schurForm(), 1e-9);
    }

    @Test void implicitDoubleShiftHandlesDenseNonsymmetricMatrix() {
        Matrix a = Matrix.of(new double[][]{
                {1, 2, 3, 4, 1},
                {5, 6, 7, 8, 2},
                {2, 6, 4, 1, 3},
                {3, 1, 8, 2, 4},
                {4, 3, 2, 1, 5}
        });
        RealSchurDecomposition result = RealSchurDecomposition.factor(a);
        assertClose(a, result.reconstruct(), 1e-7);
        assertClose(Matrix.identity(5),
                result.orthogonalQ().transpose().multiply(result.orthogonalQ()), 1e-7);
        assertQuasiTriangular(result.schurForm(), 1e-7);
    }

    @Test void triangularizesIsolatedRealPairWithOrthogonalSimilarity() {
        Matrix a = Matrix.of(new double[][]{
                {4, 0, 0},
                {0, 2, 1},
                {0, 1, 3}
        });
        RealSchurDecomposition result = RealSchurDecomposition.factor(a);
        assertClose(a, result.reconstruct(), 1e-9);
        assertClose(Matrix.identity(3),
                result.orthogonalQ().transpose().multiply(result.orthogonalQ()), 1e-9);
        assertQuasiTriangular(result.schurForm(), 1e-9);
    }

    @Test void preservesSeparatedComplexPairsInBlockDiagonalMatrix() {
        Matrix a = Matrix.of(new double[][]{
                {0, -1, 0, 0},
                {1, 0, 0, 0},
                {0, 0, 0, -2},
                {0, 0, 2, 0}
        });
        RealSchurDecomposition result = RealSchurDecomposition.factor(a);
        assertClose(a, result.reconstruct(), 1e-9);
        assertTrue(result.isComplexPairBlock(0, NumericalPrecision.FLOAT64.defaultTolerance()));
        assertTrue(result.isComplexPairBlock(2, NumericalPrecision.FLOAT64.defaultTolerance()));
    }

    @Test void rejectsNonsquareInput() {
        assertThrows(IllegalArgumentException.class, () ->
                RealSchurDecomposition.factor(Matrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}})));
    }

    private static void assertQuasiTriangular(Matrix a, double tolerance) {
        for (int i = 0; i < a.rows(); i++)
            for (int j = 0; j < i - 1; j++)
                assertEquals(0.0, a.get(i, j), tolerance);
    }

    private static void assertClose(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.rows(), actual.rows());
        assertEquals(expected.columns(), actual.columns());
        for (int i = 0; i < expected.rows(); i++)
            for (int j = 0; j < expected.columns(); j++)
                assertEquals(expected.get(i, j), actual.get(i, j), tolerance);
    }
}
