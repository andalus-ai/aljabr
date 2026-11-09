package tech.kayys.aljabr.core.nn;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;

class ParameterTest {

    @Test
    void parameterHandleDelegatesGradientStateToItsTensor() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Parameter weight = layer.namedParameterObjects().get("weight");

            assertThat(weight).isNotNull();
            assertThat(weight.name()).isEqualTo("weight");
            assertThat(weight.tensor()).isSameAs(layer.parameters().get("weight"));
            assertThat(weight.requiresGrad()).isFalse();

            weight.setRequiresGrad(true);

            assertThat(layer.parameters().get("weight").requiresGrad()).isTrue();
            assertThat(layer.trainableParameterObjects()).contains(weight);

            weight.setRequiresGrad(false);

            assertThat(layer.trainableParameterObjects()).doesNotContain(weight);
        }
    }

    @Test
    void parameterHandlesAreAvailableThroughQualifiedModuleNames() {
        CpuBackend backend = new CpuBackend();
        try (Sequential model = new Sequential(backend,
                new Linear(backend, 2, 3, true),
                new Linear(backend, 3, 1, false))) {

            assertThat(model.namedParameterObjects())
                    .containsKeys("0.weight", "0.bias", "1.weight");
            assertThat(model.parameterObjects()).hasSize(3);
        }
    }
}
