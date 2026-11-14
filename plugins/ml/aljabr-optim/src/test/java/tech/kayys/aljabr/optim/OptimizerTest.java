package tech.kayys.aljabr.optim;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class OptimizerTest {

    @Test
    void sgdUpdatesRegisteredParameterAndPreservesHandle() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, false)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Tensor original = parameter.tensor();
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            new SGD(0.1f).step(Map.of("weight", parameter), Map.of("weight", gradient));

            assertThat(parameter.tensor()).isNotSameAs(original);
            assertThat(parameter.tensor()).isSameAs(layer.parameters().get("weight"));
            assertThat(parameter.requiresGrad()).isTrue();
            assertThat(parameter.tensor().toFloatArray())
                    .containsExactlyElementsOf(java.util.Arrays.stream(original.toFloatArray())
                            .map(value -> value - 0.1f).boxed().toList());
            gradient.release();
        }
    }

    @Test
    void sgdValidatesAllGradientsBeforeReplacingAnyParameter() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            Tensor originalWeight = weight.tensor();
            Tensor invalid = TensorFactory.zeros(backend, 7);
            assertThrows(IllegalArgumentException.class, () -> new SGD(0.1f).step(
                    Map.of("weight", weight, "bias", bias),
                    Map.of("weight", TensorFactory.ones(backend, 2, 2), "bias", invalid)));
            assertThat(weight.tensor()).isSameAs(originalWeight);
            invalid.release();
        }
    }

    @Test
    void adamWUpdatesAndCanResetItsState() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, false);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.01f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            Tensor original = parameter.tensor();
            optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
            assertThat(parameter.tensor()).isNotSameAs(original);
            optimizer.reset();
            gradient.release();
        }
    }

    @Test
    void frozenParametersAreNotUpdated() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, false)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            parameter.setRequiresGrad(false);
            Tensor original = parameter.tensor();
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            new SGD(0.1f).step(Map.of("weight", parameter), Map.of("weight", gradient));
            assertThat(parameter.tensor()).isSameAs(original);
            gradient.release();
        }
    }
}
