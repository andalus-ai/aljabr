package tech.kayys.aljabr.models;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class ResnetBlock2DBackendTest {

    @Test
    void propagatesBackendToEveryChildAndStateTensor() {
        CpuBackend backend = new CpuBackend();
        try (ResnetBlock2D block = new ResnetBlock2D(backend, 32, 64, 16)) {
            assertThat(block.computeBackend()).isSameAs(backend);
            assertThat(block.children().values())
                    .allSatisfy(child -> assertThat(child.computeBackend()).isSameAs(backend));
            assertThat(block.stateDict().values())
                    .allSatisfy(tensor -> assertThat(tensor.backend()).isSameAs(backend));
        }
    }

    @Test
    void rejectsInputAndTimestepEmbeddingFromAnotherBackend() {
        CpuBackend backend = new CpuBackend();
        CpuBackend other = new CpuBackend();
        try (ResnetBlock2D block = new ResnetBlock2D(backend, 32, 32, 16)) {
            Tensor wrongInput = TensorFactory.of(new float[32], other, 1, 32, 1, 1);
            Tensor input = TensorFactory.of(new float[32], backend, 1, 32, 1, 1);
            Tensor wrongTemb = TensorFactory.of(new float[16], other, 1, 16);
            try {
                assertThatThrownBy(() -> block.forward(wrongInput, null))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("backend");
                assertThatThrownBy(() -> block.forward(input, wrongTemb))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("backend");
            } finally {
                wrongInput.release();
                input.release();
                wrongTemb.release();
            }
        }
    }

    @Test
    void rejectsInputsWithWrongChannelLayout() {
        CpuBackend backend = new CpuBackend();
        try (ResnetBlock2D block = new ResnetBlock2D(backend, 32, 32, 16)) {
            Tensor input = TensorFactory.of(new float[16], backend, 1, 16, 1, 1);
            try {
                assertThatThrownBy(() -> block.forward(input))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("expected");
            } finally {
                input.release();
            }
        }
    }
}
