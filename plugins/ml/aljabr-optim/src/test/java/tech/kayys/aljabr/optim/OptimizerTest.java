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
            SGD optimizer = new SGD(0.1f);
            optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
            // A second step verifies that the first replacement is a fresh leaf tensor.
            optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));

            assertThat(parameter.tensor()).isNotSameAs(original);
            assertThat(parameter.tensor()).isSameAs(layer.parameters().get("weight"));
            assertThat(parameter.requiresGrad()).isTrue();
            float[] expected = original.toFloatArray();
            for (int i = 0; i < expected.length; i++) {
                expected[i] -= 0.2f;
            }
            assertThat(parameter.tensor().toFloatArray()).containsExactly(expected);
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
            Tensor validWeightGradient = TensorFactory.ones(backend, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> new SGD(0.1f).step(
                    Map.of("weight", weight, "bias", bias),
                    Map.of("weight", validWeightGradient, "bias", invalid)));
            assertThat(weight.tensor()).isSameAs(originalWeight);
            validWeightGradient.release();
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
            // Optimizer arithmetic must not build a new autograd graph across steps.
            optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
            optimizer.reset();
            gradient.release();
        }
    }

    @Test
    void adamWResetsMomentsWhenRegisteredParameterTensorIsReplaced() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, false);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Tensor positiveGradient = TensorFactory.ones(backend, 2, 2);
            optimizer.step(Map.of("weight", parameter), Map.of("weight", positiveGradient));

            Tensor replacement = TensorFactory.ones(backend, 2, 2);
            parameter.updateTensor(replacement);
            Tensor negativeGradient = TensorFactory.ones(backend, 2, 2).mul(-1.0f);
            optimizer.step(Map.of("weight", parameter), Map.of("weight", negativeGradient));

            // A fresh first moment for a negative gradient must move the parameter upward.
            for (float value : parameter.tensor().toFloatArray()) {
                assertThat(value).isGreaterThan(1.0f);
            }
            positiveGradient.release();
            negativeGradient.release();
        }
    }

    @Test
    void trainsLinearAcrossConsecutiveForwardBackwardAndOptimizerSteps() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 3f, 4f }, backend, 1, 2);
        Tensor initialWeight = TensorFactory.of(new float[] { 1f, 2f }, backend, 1, 2);
        try (Linear layer = new Linear(initialWeight, null)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            SGD optimizer = new SGD(0.1f);

            try (Tensor output = layer.forward(input);
                 Tensor loss = output.sum()) {
                loss.backward();
            }
            assertThat(parameter.tensor().grad().toFloatArray()).containsExactly(3f, 4f);
            optimizer.step(Map.of("weight", parameter),
                    Map.of("weight", parameter.tensor().grad()));
            assertThat(parameter.tensor().toFloatArray()).containsExactly(0.7f, 1.6f);

            try (Tensor output = layer.forward(input);
                 Tensor loss = output.sum()) {
                loss.backward();
            }
            assertThat(parameter.tensor().grad().toFloatArray()).containsExactly(3f, 4f);
            optimizer.step(Map.of("weight", parameter),
                    Map.of("weight", parameter.tensor().grad()));
            assertThat(parameter.tensor().toFloatArray()).containsExactly(0.4f, 1.2f);
        } finally {
            input.release();
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
