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

    @Test void randomizedDenseMatricesPreserveSimilarityAndOrthogonality() {
        java.util.Random random = new java.util.Random(0x5EEDC0DEL);
        for (int order = 3; order <= 8; order++) {
            for (int sample = 0; sample < 4; sample++) {
                double[][] values = new double[order][order];
                for (int row = 0; row < order; row++)
                    for (int column = 0; column < order; column++)
                        values[row][column] = random.nextDouble() * 2.0 - 1.0;
                Matrix input = Matrix.of(values);
                RealSchurDecomposition result = RealSchurDecomposition.factor(input);
                assertCloseRelative(input, result.reconstruct(), 2e-7);
                assertCloseRelative(Matrix.identity(order),
                        result.orthogonalQ().transpose().multiply(result.orthogonalQ()), 2e-7);
                assertQuasiTriangularRelative(result.schurForm(), 2e-7);
            }
        }
    }

    @Test void remainsStableAcrossSmallAndLargeUniformScales() {
        Matrix base = Matrix.of(new double[][]{
                {2, -1, 3, 0},
                {4, 1, -2, 5},
                {-3, 2, 0, 1},
                {1, 4, 2, -1}
        });
        for (double scale : new double[]{1e-8, 1e-4, 1e4, 1e8}) {
            Matrix input = base.scale(scale);
            RealSchurDecomposition result = RealSchurDecomposition.factor(input);
            assertCloseRelative(input, result.reconstruct(), 5e-7);
            assertCloseRelative(Matrix.identity(4),
                    result.orthogonalQ().transpose().multiply(result.orthogonalQ()), 5e-7);
            assertQuasiTriangularRelative(result.schurForm(), 5e-7);
        }
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

    private static void assertQuasiTriangularRelative(Matrix a, double tolerance) {
        double scale = Math.max(Double.MIN_NORMAL, a.normInfinity());
        for (int i = 0; i < a.rows(); i++)
            for (int j = 0; j < i - 1; j++)
                assertEquals(0.0, a.get(i, j), tolerance * scale);
    }

    private static void assertCloseRelative(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.rows(), actual.rows());
        assertEquals(expected.columns(), actual.columns());
        double scale = Math.max(Double.MIN_NORMAL, expected.normInfinity());
        for (int i = 0; i < expected.rows(); i++)
            for (int j = 0; j < expected.columns(); j++)
                assertEquals(expected.get(i, j), actual.get(i, j), tolerance * scale);
    }

    private static void assertClose(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.rows(), actual.rows());
        assertEquals(expected.columns(), actual.columns());
        for (int i = 0; i < expected.rows(); i++)
            for (int j = 0; j < expected.columns(); j++)
                assertEquals(expected.get(i, j), actual.get(i, j), tolerance);
    }
}
