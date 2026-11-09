package tech.kayys.aljabr.models;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class SpatialTransformerBackendTest {

    @Test
    void propagatesBackendToEveryChildAndStateTensor() {
        CpuBackend backend = new CpuBackend();
        try (SpatialTransformer transformer = new SpatialTransformer(backend, 32, 2, 8, 12)) {
            assertThat(transformer.computeBackend()).isSameAs(backend);
            assertThat(transformer.children().values())
                    .allSatisfy(child -> assertThat(child.computeBackend()).isSameAs(backend));
            assertThat(transformer.stateDict().values())
                    .allSatisfy(tensor -> assertThat(tensor.backend()).isSameAs(backend));
            assertThat(transformer.innerDim()).isEqualTo(16);
            assertThat(transformer.contextDim()).isEqualTo(12);
        }
    }

    @Test
    void rejectsInputAndContextFromAnotherBackend() {
        CpuBackend backend = new CpuBackend();
        CpuBackend other = new CpuBackend();
        try (SpatialTransformer transformer = new SpatialTransformer(backend, 32, 2, 8, 12)) {
            Tensor wrongInput = TensorFactory.of(new float[32], other, 1, 32, 1, 1);
            Tensor wrongContext = TensorFactory.of(new float[36], other, 1, 3, 12);
            Tensor input = TensorFactory.of(new float[32], backend, 1, 32, 1, 1);
            try {
                assertThatThrownBy(() -> transformer.forward(wrongInput, null))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("backend");
                assertThatThrownBy(() -> transformer.forward(input, wrongContext))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("backend");
            } finally {
                wrongInput.release();
                wrongContext.release();
                input.release();
            }
        }
    }

    @Test
    void rejectsInvalidSpatialAndContextShapes() {
        CpuBackend backend = new CpuBackend();
        try (SpatialTransformer transformer = new SpatialTransformer(backend, 32, 2, 8, 12)) {
            Tensor wrongChannels = TensorFactory.of(new float[16], backend, 1, 16, 1, 1);
            Tensor input = TensorFactory.of(new float[32], backend, 1, 32, 1, 1);
            Tensor wrongContext = TensorFactory.of(new float[24], backend, 1, 2, 12);
            try {
                assertThatThrownBy(() -> transformer.forward(wrongChannels, null))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("expected");
                assertThatThrownBy(() -> transformer.forward(input, wrongContext))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("context");
            } finally {
                wrongChannels.release();
                input.release();
                wrongContext.release();
            }
        }
    }

    @Test
    void rejectsInvalidArchitectureDimensions() {
        CpuBackend backend = new CpuBackend();
        assertThatThrownBy(() -> new SpatialTransformer(backend, 30, 2, 8, 12))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("GroupNorm");
        assertThatThrownBy(() -> new SpatialTransformer(backend, 32, 0, 8, 12))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("positive");
    }
}
