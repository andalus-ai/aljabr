package tech.kayys.aljabr.models;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class UNet2DConditionalBackendTest {

    @Test
    void propagatesBackendToTopLevelModulesAndParameters() {
        CpuBackend backend = new CpuBackend();
        try (UNet2DConditional model = new UNet2DConditional(backend, 4, 4, 8, 12)) {
            assertThat(model.computeBackend()).isSameAs(backend);
            assertThat(model.children().values())
                    .allSatisfy(child -> assertThat(child.computeBackend()).isSameAs(backend));
            assertThat(model.stateDict().values())
                    .allSatisfy(tensor -> assertThat(tensor.backend()).isSameAs(backend));
            assertThat(model.inChannels()).isEqualTo(4);
            assertThat(model.outChannels()).isEqualTo(4);
        }
    }

    @Test
    void rejectsInputsOwnedByAnotherBackendBeforeExecution() {
        CpuBackend backend = new CpuBackend();
        CpuBackend other = new CpuBackend();
        try (UNet2DConditional model = new UNet2DConditional(backend, 4, 4, 8, 12)) {
            Tensor latents = TensorFactory.of(new float[16], other, 1, 4, 2, 2);
            Tensor timestep = TensorFactory.of(new float[8], backend, 1, 8);
            try {
                assertThatThrownBy(() -> model.forward(latents, timestep, null))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("backend");
            } finally {
                latents.release();
                timestep.release();
            }
        }
    }

    @Test
    void validatesLatentTimestepAndContextContracts() {
        CpuBackend backend = new CpuBackend();
        try (UNet2DConditional model = new UNet2DConditional(backend, 4, 4, 8, 12)) {
            Tensor latents = TensorFactory.of(new float[16], backend, 1, 4, 2, 2);
            Tensor wrongLatents = TensorFactory.of(new float[12], backend, 1, 3, 2, 2);
            Tensor wrongTimestep = TensorFactory.of(new float[7], backend, 1, 7);
            Tensor wrongContext = TensorFactory.of(new float[26], backend, 1, 2, 13);
            try {
                assertThatThrownBy(() -> model.forward(wrongLatents, wrongTimestep, null))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("latents");
                Tensor timestep = TensorFactory.of(new float[8], backend, 1, 8);
                try {
                    assertThatThrownBy(() -> model.forward(latents, wrongTimestep, null))
                            .isInstanceOf(IllegalArgumentException.class)
                            .hasMessageContaining("timestep");
                    assertThatThrownBy(() -> model.forward(latents, timestep, wrongContext))
                            .isInstanceOf(IllegalArgumentException.class)
                            .hasMessageContaining("context");
                } finally {
                    timestep.release();
                }
            } finally {
                latents.release();
                wrongLatents.release();
                wrongTimestep.release();
                wrongContext.release();
            }
        }
    }

    @Test
    void rejectsInvalidModelDimensions() {
        CpuBackend backend = new CpuBackend();
        assertThatThrownBy(() -> new UNet2DConditional(backend, 0, 4, 8, 12))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("positive");
    }
}
