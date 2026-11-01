package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class HessenbergReductionTest {
    @Test void reducesNonsymmetricMatrixAndReconstructsInput() {
        Matrix a = Matrix.of(new double[][]{
                {4, 1, -2, 2},
                {1, 2, 0, 1},
                {-2, 0, 3, -2},
                {2, 1, -2, -1}
        });
        HessenbergReduction reduction = HessenbergReduction.reduce(a);
        Matrix h = reduction.hessenberg();
        Matrix q = reduction.orthogonalQ();

        assertMatrixClose(a, reduction.reconstruct(), 1e-9);
        assertMatrixClose(Matrix.identity(4), q.transpose().multiply(q), 1e-9);
        for (int i = 0; i < h.rows(); i++)
            for (int j = 0; j < i - 1; j++)
                assertEquals(0.0, h.get(i, j), 0.0);
    }

    @Test void handlesSmallAndAlreadyHessenbergMatrices() {
        Matrix twoByTwo = Matrix.of(new double[][]{{1, 2}, {3, 4}});
        HessenbergReduction small = HessenbergReduction.reduce(twoByTwo);
        assertMatrixClose(twoByTwo, small.hessenberg(), 1e-12);
        assertMatrixClose(Matrix.identity(2), small.orthogonalQ(), 1e-12);

        Matrix h = Matrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}, {0, 7, 8}});
        HessenbergReduction unchanged = HessenbergReduction.reduce(h);
        assertMatrixClose(h, unchanged.hessenberg(), 1e-12);
    }

    @Test void symmetricInputReducesToTridiagonalForm() {
        Matrix a = Matrix.of(new double[][]{
                {4, 1, 2, 0},
                {1, 3, 0, 1},
                {2, 0, 2, 1},
                {0, 1, 1, 5}
        });
        Matrix h = HessenbergReduction.reduce(a).hessenberg();
        for (int i = 0; i < h.rows(); i++)
            for (int j = 0; j < i - 1; j++)
                assertEquals(0.0, h.get(i, j), 0.0);
        for (int i = 0; i < h.rows(); i++)
            for (int j = i + 2; j < h.columns(); j++)
                assertEquals(0.0, h.get(i, j), 1e-9);
    }

    @Test void rejectsNonsquareMatrices() {
        assertThrows(IllegalArgumentException.class, () ->
                HessenbergReduction.reduce(Matrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}})));
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
