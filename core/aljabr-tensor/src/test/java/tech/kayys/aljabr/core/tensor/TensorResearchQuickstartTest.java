package tech.kayys.aljabr.core.tensor;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Executable counterpart to docs/examples/TENSOR_RESEARCH_QUICKSTART.md.
 */
class TensorResearchQuickstartTest {

    @Test
    void matrixMultiplicationAndMeanMatchReferenceValues() {
        Tensor inputs = Tensor.of(new float[] {
                1.0f, 2.0f,
                3.0f, 4.0f
        }, 2, 2);

        Tensor weights = Tensor.of(new float[] {
                0.5f, 1.0f,
                1.5f, 2.0f
        }, 2, 2);

        Tensor predictions = null;
        Tensor mean = null;
        try {
            predictions = inputs.matmul(weights);
            mean = predictions.mean();

            assertThat(predictions.shape().dims()).containsExactly(2, 2);
            assertThat(predictions.toFloatArray()).containsExactly(3.5f, 5.0f, 7.5f, 11.0f);
            assertThat(mean.item()).isEqualTo(6.75f);
        } finally {
            if (mean != null) {
                mean.release();
            }
            if (predictions != null) {
                predictions.release();
            }
            weights.release();
            inputs.release();
        }
    }
}
