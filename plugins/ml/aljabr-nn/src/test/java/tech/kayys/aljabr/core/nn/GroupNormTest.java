package tech.kayys.aljabr.core.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class GroupNormTest {

    @Test
    void normalizesEachChannelGroupIndependently() {
        CpuBackend backend = new CpuBackend();
        GroupNorm layer = new GroupNorm(backend, 2, 4, 1e-5f, false);
        Tensor input = TensorFactory.of(new float[] { 1f, 3f, 5f, 7f }, backend, 1, 4, 1, 1);
        try {
            Tensor output = layer.forward(input);
            try {
                assertThat(output.shape().dims()).containsExactly(1, 4, 1, 1);
                assertThat(output.backend()).isSameAs(backend);
                float[] values = output.toFloatArray();
                assertThat(values[0]).isCloseTo(-0.999995f, within(2e-5f));
                assertThat(values[1]).isCloseTo(0.999995f, within(2e-5f));
                assertThat(values[2]).isCloseTo(-0.999995f, within(2e-5f));
                assertThat(values[3]).isCloseTo(0.999995f, within(2e-5f));
            } finally {
                output.release();
            }
        } finally {
            input.release();
            layer.close();
        }
    }

    @Test
    void affineParametersDefaultToIdentity() {
        CpuBackend backend = new CpuBackend();
        GroupNorm layer = new GroupNorm(backend, 1, 2);
        Tensor input = TensorFactory.of(new float[] { 2f, 4f }, backend, 1, 2);
        try {
            Tensor output = layer.forward(input);
            try {
                float[] values = output.toFloatArray();
                assertThat(values[0]).isCloseTo(-0.999995f, within(2e-5f));
                assertThat(values[1]).isCloseTo(0.999995f, within(2e-5f));
            } finally {
                output.release();
            }
        } finally {
            input.release();
            layer.close();
        }
    }

    @Test
    void rejectsInvalidChannelShapeAndBackend() {
        CpuBackend backend = new CpuBackend();
        GroupNorm layer = new GroupNorm(backend, 2, 4);
        Tensor wrongChannels = TensorFactory.of(new float[3], backend, 1, 3);
        Tensor wrongBackend = TensorFactory.of(new float[4], new CpuBackend(), 1, 4);
        try {
            assertThatThrownBy(() -> layer.forward(wrongChannels))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("channels");
            assertThatThrownBy(() -> layer.forward(wrongBackend))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("backend");
        } finally {
            wrongChannels.release();
            wrongBackend.release();
            layer.close();
        }
    }

    @Test
    void rejectsNonDivisibleGroupConfiguration() {
        assertThatThrownBy(() -> new GroupNorm(new CpuBackend(), 3, 4))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("divisible");
    }
}
