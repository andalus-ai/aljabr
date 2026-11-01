package tech.kayys.aljabr.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ConditionNumberTest {
    @Test void identityHasUnitConditionNumber() {
        ConditionNumber condition = ConditionNumber.of(Matrix.identity(3));
        assertEquals(1.0, condition.twoNorm(), 1e-12);
        assertEquals(1.0, condition.reciprocalTwoNorm(), 1e-12);
        assertEquals(3, condition.numericalRank());
        assertFalse(condition.isNumericallyRankDeficient());
    }

    @Test void diagonalMatrixUsesLargestOverSmallestSingularValue() {
        ConditionNumber condition = ConditionNumber.of(
                Matrix.of(new double[][]{{4, 0}, {0, 2}}));
        assertEquals(2.0, condition.twoNorm(), 1e-10);
        assertEquals(0.5, condition.reciprocalTwoNorm(), 1e-10);
    }

    @Test void rankDeficientMatrixHasInfiniteConditionNumber() {
        ConditionNumber condition = ConditionNumber.of(
                Matrix.of(new double[][]{{1, 2}, {2, 4}, {3, 6}}));
        assertTrue(Double.isInfinite(condition.twoNorm()));
        assertEquals(0.0, condition.reciprocalTwoNorm());
        assertEquals(1, condition.numericalRank());
        assertEquals(2, condition.maximumRank());
        assertTrue(condition.isNumericallyRankDeficient());
    }

    @Test void rectangularFullRankMatrixIsSupported() {
        ConditionNumber condition = ConditionNumber.of(
                Matrix.of(new double[][]{{1, 0}, {0, 2}, {0, 0}}));
        assertEquals(2.0, condition.twoNorm(), 1e-10);
        assertEquals(2, condition.numericalRank());
        assertEquals(2, condition.maximumRank());
    }

    @Test void zeroMatrixIsRankDeficient() {
        ConditionNumber condition = ConditionNumber.of(
                Matrix.of(new double[][]{{0, 0}, {0, 0}}));
        assertTrue(Double.isInfinite(condition.twoNorm()));
        assertEquals(0, condition.numericalRank());
    }
}
