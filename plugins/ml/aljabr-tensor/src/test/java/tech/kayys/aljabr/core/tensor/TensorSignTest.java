package tech.kayys.aljabr.core.tensor;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;

class TensorSignTest {

    @Test
    void signPreservesShapeAndMapsFiniteValuesAndSpecialValues() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor input = TensorFactory.of(
                    new float[] { -3.0f, -0.0f, 0.0f, 2.5f, Float.NEGATIVE_INFINITY,
                            Float.POSITIVE_INFINITY, Float.NaN },
                    backend, 1, 7);
            Tensor output = null;
            try {
                output = input.sign();
                assertThat(output.shape().dims()).containsExactly(1, 7);
                float[] values = output.toFloatArray();
                assertThat(values[0]).isEqualTo(-1.0f);
                assertThat(values[1]).isEqualTo(-0.0f);
                assertThat(values[2]).isEqualTo(0.0f);
                assertThat(values[3]).isEqualTo(1.0f);
                assertThat(values[4]).isEqualTo(-1.0f);
                assertThat(values[5]).isEqualTo(1.0f);
                assertThat(values[6]).isNaN();
            } finally {
                if (output != null) output.release();
                input.release();
            }
        }
    }
}
