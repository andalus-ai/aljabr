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
            assertThat(weight.requiresGrad()).isTrue();
            assertThat(layer.trainableParameterObjects()).contains(weight);

            weight.setRequiresGrad(false);

            assertThat(layer.parameters().get("weight").requiresGrad()).isFalse();
            assertThat(layer.trainableParameterObjects()).doesNotContain(weight);

            weight.setRequiresGrad(true);

            assertThat(layer.parameters().get("weight").requiresGrad()).isTrue();
            assertThat(layer.trainableParameterObjects()).contains(weight);
        }
    }

    @Test
    void updateTensorReplacesValueAndPreservesTrainability() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            weight.setRequiresGrad(false);
            var original = weight.tensor();
            var replacement = original.zerosLike();

            weight.updateTensor(replacement);

            assertThat(weight.tensor()).isSameAs(replacement).isNotSameAs(original);
            assertThat(weight.requiresGrad()).isFalse();
            assertThat(layer.parameters().get("weight")).isSameAs(replacement);
        }
    }

    @Test
    void updateTensorRejectsIncompatibleShape() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            var incompatible = tech.kayys.aljabr.core.tensor.TensorFactory.zeros(backend, 3, 2);

            assertThat(org.junit.jupiter.api.Assertions.assertThrows(
                    IllegalArgumentException.class, () -> weight.updateTensor(incompatible)))
                    .hasMessageContaining("shape cannot change");
            incompatible.release();
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
