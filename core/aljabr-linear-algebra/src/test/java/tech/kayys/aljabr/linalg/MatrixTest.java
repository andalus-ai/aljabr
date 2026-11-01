package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {
    @Test void matrixMultiplicationAndTranspose() {
        Matrix a = Matrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix b = Matrix.of(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        Matrix product = a.multiply(b);
        assertEquals(58.0, product.get(0, 0), 1e-12);
        assertEquals(154.0, product.get(1, 1), 1e-12);
        assertEquals(2, a.transpose().rows());
        assertEquals(3, a.transpose().columns());
    }

    @Test void copiesInputAndOutputArrays() {
        double[][] source = {{1, 2}, {3, 4}};
        Matrix matrix = Matrix.of(source);
        source[0][0] = 99;
        double[][] exported = matrix.toArray();
        exported[0][0] = 88;
        assertEquals(1.0, matrix.get(0, 0));
    }

    @Test void rejectsRaggedNonFiniteAndNonconformingValues() {
        assertThrows(IllegalArgumentException.class, () -> Matrix.of(new double[][]{{Double.NaN}}));
        assertThrows(IllegalArgumentException.class, () -> Vector.of(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> Matrix.of(new double[][]{{1}, {2, 3}}));
        assertThrows(IllegalArgumentException.class,
                () -> Matrix.of(new double[][]{{1, 2}}).multiply(Matrix.of(new double[][]{{1, 2}})));
    }
}
