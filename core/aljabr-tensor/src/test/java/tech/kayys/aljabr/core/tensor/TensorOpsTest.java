package tech.kayys.aljabr.core.tensor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;

import org.junit.jupiter.api.Test;

class TensorOpsTest {

    @Test
    void catConcatenatesAlongFirstDimension() {
        Tensor a = Tensor.of(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor b = Tensor.of(new float[] { 5, 6 }, 1, 2);
        try {
            Tensor result = TensorOps.cat(List.of(a, b), 0);
            try {
                assertThat(result.shape().dims()).containsExactly(3, 2);
                assertThat(result.toFloatArray()).containsExactly(1, 2, 3, 4, 5, 6);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void catConcatenatesAlongInnerDimensionAndSupportsNegativeAxis() {
        Tensor a = Tensor.of(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor b = Tensor.of(new float[] { 5, 6, 7, 8 }, 2, 2);
        try {
            Tensor result = TensorOps.cat(List.of(a, b), -1);
            try {
                assertThat(result.shape().dims()).containsExactly(2, 4);
                assertThat(result.toFloatArray()).containsExactly(1, 2, 5, 6, 3, 4, 7, 8);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void catRejectsEmptyListInvalidAxisAndIncompatibleShapes() {
        assertThatThrownBy(() -> TensorOps.cat(List.of(), 0))
                .isInstanceOf(IllegalArgumentException.class);
        Tensor a = Tensor.of(new float[] { 1, 2 }, 1, 2);
        Tensor wrongShape = Tensor.of(new float[] { 3, 4, 5 }, 1, 3);
        try {
            assertThatThrownBy(() -> TensorOps.cat(List.of(a), 2))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("out of range");
            assertThatThrownBy(() -> TensorOps.cat(List.of(a, wrongShape), 0))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("shape mismatch");
        } finally {
            a.release();
            aVector.release();
            wrongShape.release();
        }
    }

    @Test
    void einsumComputesMatrixMultiplication() {
        Tensor a = Tensor.of(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor b = Tensor.of(new float[] { 5, 6, 7, 8 }, 2, 2);
        try {
            Tensor result = TensorOps.einsum("ij,jk->ik", a, b);
            try {
                assertThat(result.shape().dims()).containsExactly(2, 2);
                assertThat(result.toFloatArray()).containsExactly(19, 22, 43, 50);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void einsumComputesDotProductAndScalarReduction() {
        Tensor a = Tensor.of(new float[] { 1, 2, 3 }, 3);
        Tensor b = Tensor.of(new float[] { 4, 5, 6 }, 3);
        try {
            Tensor dot = TensorOps.einsum("i,i->", a, b);
            try {
                assertThat(dot.shape().dims()).isEmpty();
                assertThat(dot.item()).isEqualTo(32.0f);
            } finally {
                dot.release();
            }

            Tensor sum = TensorOps.einsum("i->", a);
            try {
                assertThat(sum.item()).isEqualTo(6.0f);
            } finally {
                sum.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void einsumSupportsTransposeAndOuterProduct() {
        Tensor matrix = Tensor.of(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor left = Tensor.of(new float[] { 1, 2 }, 2);
        Tensor right = Tensor.of(new float[] { 3, 4, 5 }, 3);
        try {
            Tensor transpose = TensorOps.einsum("ij->ji", matrix);
            try {
                assertThat(transpose.shape().dims()).containsExactly(2, 2);
                assertThat(transpose.toFloatArray()).containsExactly(1, 3, 2, 4);
            } finally {
                transpose.release();
            }

            Tensor outer = TensorOps.einsum("i,j->ij", left, right);
            try {
                assertThat(outer.shape().dims()).containsExactly(2, 3);
                assertThat(outer.toFloatArray()).containsExactly(3, 4, 5, 6, 8, 10);
            } finally {
                outer.release();
            }
        } finally {
            matrix.release();
            left.release();
            right.release();
        }
    }

    @Test
    void einsumRejectsMalformedEquationsAndIncompatibleDimensions() {
        Tensor a = Tensor.of(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor aVector = Tensor.of(new float[] { 1, 2, 3, 4 }, 4);
        Tensor wrongShape = Tensor.of(new float[] { 1, 2, 3 }, 3);
        try {
            assertThatThrownBy(() -> TensorOps.einsum("ij,jk", a, a))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("explicit");
            assertThatThrownBy(() -> TensorOps.einsum("ii->i", a))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("repeated label");
            assertThatThrownBy(() -> TensorOps.einsum("i,i->i", aVector, wrongShape))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("dimension mismatch");
            assertThatThrownBy(() -> TensorOps.einsum("i->i", a))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("rank");
        } finally {
            a.release();
            wrongShape.release();
        }
    }
}
