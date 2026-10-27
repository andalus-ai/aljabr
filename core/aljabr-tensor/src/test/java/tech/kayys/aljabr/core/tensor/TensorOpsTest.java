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
            wrongShape.release();
        }
    }

    @Test
    void einsumFailsExplicitlyInsteadOfReturningAnInputAsAFalseResult() {
        Tensor input = Tensor.of(new float[] { 1, 2 }, 2);
        try {
            assertThatThrownBy(() -> TensorOps.einsum("i->i", input))
                    .isInstanceOf(UnsupportedOperationException.class)
                    .hasMessageContaining("not implemented");
        } finally {
            input.release();
        }
    }
}
