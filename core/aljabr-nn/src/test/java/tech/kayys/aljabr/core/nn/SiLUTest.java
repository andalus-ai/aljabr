package tech.kayys.aljabr.core.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class SiLUTest {

    @Test
    void appliesSiluOnItsConfiguredBackend() {
        CpuBackend backend = new CpuBackend();
        SiLU activation = new SiLU(backend);
        Tensor input = TensorFactory.of(new float[] { -2f, 0f, 2f }, backend, 3);
        try {
            Tensor output = activation.forward(input);
            try {
                assertThat(output.backend()).isSameAs(backend);
                float[] values = output.toFloatArray();
                assertThat(values[0]).isCloseTo(-2f / (1f + (float) Math.exp(2f)), within(1e-6f));
                assertThat(values[1]).isCloseTo(0f, within(1e-6f));
                assertThat(values[2]).isCloseTo(2f / (1f + (float) Math.exp(-2f)), within(1e-6f));
            } finally {
                output.release();
            }
        } finally {
            input.release();
        }
    }

    @Test
    void rejectsInputOwnedByAnotherBackend() {
        SiLU activation = new SiLU(new CpuBackend());
        Tensor input = TensorFactory.of(new float[] { 1f }, new CpuBackend(), 1);
        try {
            assertThatThrownBy(() -> activation.forward(input))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("backend");
        } finally {
            input.release();
        }
    }
}
