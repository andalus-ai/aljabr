package tech.kayys.aljabr.optim;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.assertj.core.data.Offset.offset;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class OptimizerTest {

    @Test
    void optimizersRejectInvalidHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new SGD(-0.01f));
        assertThrows(IllegalArgumentException.class, () -> new SGD(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new SGD(Float.POSITIVE_INFINITY));

        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(-0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(Float.NaN, 0.9f, 0.999f, 1.0e-8f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, -0.1f, 0.999f, 1.0e-8f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, 1.0f, 0.999f, 1.0e-8f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, 0.9f, Float.NaN, 1.0e-8f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, 0.9f, 1.0f, 1.0e-8f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, 0.9f, 0.999f, 0.0f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, 0.9f, 0.999f, Float.POSITIVE_INFINITY, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, -0.01f));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, Float.NaN));
    }

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
            float[] actual = parameter.tensor().toFloatArray();
            for (int i = 0; i < expected.length; i++) {
                assertThat(actual[i]).isCloseTo(expected[i], offset(1.0e-6f));
            }
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
             Linear referenceLayer = new Linear(backend, 2, 2, false);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f);
             AdamW referenceOptimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Parameter referenceParameter = referenceLayer.namedParameterObjects().get("weight");
            Tensor positiveGradient = TensorFactory.ones(backend, 2, 2);
            Tensor replacement = TensorFactory.ones(backend, 2, 2);
            Tensor negativeGradient = TensorFactory.full(-1.0f, backend, 2, 2);
            Tensor referenceInitialValue = TensorFactory.ones(backend, 2, 2);
            Tensor referenceGradient = TensorFactory.full(-1.0f, backend, 2, 2);
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", positiveGradient));

                parameter.updateTensor(replacement);
                referenceParameter.updateTensor(referenceInitialValue);
                optimizer.step(Map.of("weight", parameter), Map.of("weight", negativeGradient));
                referenceOptimizer.step(
                        Map.of("weight", referenceParameter), Map.of("weight", referenceGradient));

                // Replacing a parameter must restart AdamW exactly like a fresh optimizer step.
                assertThat(parameter.tensor().toFloatArray())
                        .containsExactly(referenceParameter.tensor().toFloatArray());
            } finally {
                positiveGradient.release();
                negativeGradient.release();
                referenceGradient.release();
            }
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
            assertThat(parameter.tensor().toFloatArray()[0]).isBetween(0.6999f, 0.7001f);
            assertThat(parameter.tensor().toFloatArray()[1]).isBetween(1.5999f, 1.6001f);

            try (Tensor output = layer.forward(input);
                 Tensor loss = output.sum()) {
                loss.backward();
            }
            assertThat(parameter.tensor().grad().toFloatArray()).containsExactly(3f, 4f);
            optimizer.step(Map.of("weight", parameter),
                    Map.of("weight", parameter.tensor().grad()));
            assertThat(parameter.tensor().toFloatArray()[0]).isBetween(0.3999f, 0.4001f);
            assertThat(parameter.tensor().toFloatArray()[1]).isBetween(1.1999f, 1.2001f);
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

    @Test
    void adamWUpdatesEveryTrainableParameterInOneStep() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.01f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            Tensor originalWeight = weight.tensor();
            Tensor originalBias = bias.tensor();
            Tensor weightGradient = TensorFactory.ones(backend, 2, 2);
            Tensor biasGradient = TensorFactory.ones(backend, 2);
            try {
                optimizer.step(
                        Map.of("weight", weight, "bias", bias),
                        Map.of("weight", weightGradient, "bias", biasGradient));

                assertThat(weight.tensor()).isNotSameAs(originalWeight);
                assertThat(bias.tensor()).isNotSameAs(originalBias);
                assertThat(weight.tensor()).isSameAs(layer.parameters().get("weight"));
                assertThat(bias.tensor()).isSameAs(layer.parameters().get("bias"));
                assertThat(weight.tensor().toFloatArray()).isNotEqualTo(originalWeight.toFloatArray());
                assertThat(bias.tensor().toFloatArray()).isNotEqualTo(originalBias.toFloatArray());
            } finally {
                weightGradient.release();
                biasGradient.release();
            }
        }
    }

    @Test
    void adamWResetRestartsBiasCorrectionFromCurrentParameterValues() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, false);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                optimizer.reset();

                Tensor resetValue = TensorFactory.of(
                        parameter.tensor().toFloatArray(), backend, 2, 2);
                try (Linear referenceLayer = new Linear(resetValue, null);
                     AdamW freshOptimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f)) {
                    Parameter referenceParameter = referenceLayer.namedParameterObjects().get("weight");
                    freshOptimizer.step(
                            Map.of("weight", referenceParameter),
                            Map.of("weight", gradient));

                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray())
                            .containsExactly(referenceParameter.tensor().toFloatArray());
                }
            } finally {
                gradient.release();
            }
        }
    }


    @Test
    void adamWSkipsFrozenParametersWhileUpdatingTrainableParameters() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.01f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            bias.setRequiresGrad(false);
            Tensor originalWeight = weight.tensor();
            Tensor originalBias = bias.tensor();
            Tensor weightGradient = TensorFactory.ones(backend, 2, 2);
            Tensor biasGradient = TensorFactory.ones(backend, 2);
            try {
                optimizer.step(
                        Map.of("weight", weight, "bias", bias),
                        Map.of("weight", weightGradient, "bias", biasGradient));

                assertThat(weight.tensor()).isNotSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);
            } finally {
                weightGradient.release();
                biasGradient.release();
            }
        }
    }

    @Test
    void adamWMissingGradientDoesNotAdvanceOptimizerState() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, false);
             Linear referenceLayer = new Linear(backend, 2, 2, false);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f);
             AdamW referenceOptimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Parameter referenceParameter = referenceLayer.namedParameterObjects().get("weight");
            Tensor referenceInitialValue = TensorFactory.of(
                    parameter.tensor().toFloatArray(), backend, 2, 2);
            referenceParameter.updateTensor(referenceInitialValue);
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            try {
                optimizer.step(Map.of("weight", parameter), Map.of());

                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                referenceOptimizer.step(
                        Map.of("weight", referenceParameter), Map.of("weight", gradient));

                assertThat(parameter.tensor().toFloatArray())
                        .containsExactly(referenceParameter.tensor().toFloatArray());
            } finally {
                gradient.release();
            }
        }
    }

    @Test
    void adamWAppliesDecoupledWeightDecayWhenGradientIsZero() {
        CpuBackend backend = new CpuBackend();
        Tensor initial = TensorFactory.of(new float[] { 2.0f, -4.0f }, backend, 1, 2);
        try (Linear layer = new Linear(initial, null);
             AdamW optimizer = new AdamW(0.1f, 0.9f, 0.999f, 1.0e-8f, 0.2f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Tensor zeroGradient = TensorFactory.zeros(backend, 1, 2);
            try {
                optimizer.step(
                        Map.of("weight", parameter),
                        Map.of("weight", zeroGradient));

                // AdamW applies (1 - learningRate * weightDecay) independently of the gradient.
                float[] actual = parameter.tensor().toFloatArray();
                assertThat(actual[0]).isCloseTo(1.96f, offset(1.0e-6f));
                assertThat(actual[1]).isCloseTo(-3.92f, offset(1.0e-6f));
            } finally {
                zeroGradient.release();
            }
        }
    }

    @Test
    void adamWValidatesEveryGradientBeforeUpdatingAnyParameter() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true);
             AdamW optimizer = new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.01f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            Tensor originalWeight = weight.tensor();
            Tensor originalBias = bias.tensor();
            Tensor validWeightGradient = TensorFactory.ones(backend, 2, 2);
            Tensor invalidBiasGradient = TensorFactory.zeros(backend, 3);
            try {
                assertThrows(IllegalArgumentException.class, () -> optimizer.step(
                        Map.of("weight", weight, "bias", bias),
                        Map.of("weight", validWeightGradient, "bias", invalidBiasGradient)));

                assertThat(weight.tensor()).isSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);
            } finally {
                validWeightGradient.release();
                invalidBiasGradient.release();
            }
        }
    }

    @Test
    void sgdMatchesIndependentTwoStepScalarReference() {
        CpuBackend backend = new CpuBackend();
        Tensor initial = TensorFactory.of(new float[] { 1.5f }, backend, 1, 1);
        Tensor firstGradient = TensorFactory.of(new float[] { 0.25f }, backend, 1, 1);
        Tensor secondGradient = TensorFactory.of(new float[] { -0.5f }, backend, 1, 1);
        try (Linear layer = new Linear(initial, null)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            SGD optimizer = new SGD(0.1f);
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(1.475f, offset(1.0e-6f));

                optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                // 1.5 - (0.1 * 0.25) - (0.1 * -0.5) = 1.525.
                assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(1.525f, offset(1.0e-6f));
            } finally {
                firstGradient.release();
                secondGradient.release();
            }
        }
    }

    @Test
    void adamWMatchesIndependentTwoStepBiasCorrectedScalarReference() {
        CpuBackend backend = new CpuBackend();
        Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
        Tensor firstGradient = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
        Tensor secondGradient = TensorFactory.zeros(backend, 1, 1);
        try (Linear layer = new Linear(initial, null);
             AdamW optimizer = new AdamW(0.1f, 0.5f, 0.5f, 1.0e-8f, 0.0f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                assertThat(parameter.tensor().toFloatArray()[0])
                        .isCloseTo(0.9f, offset(2.0e-6f));

                optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                // Scalar reference: m=(0.5, 0.25), v=(0.5, 0.25).
                // Bias-corrected step two gives mHat=vHat=1/3, so the
                // second update is 0.1 * (1/3) / (sqrt(1/3) + 1e-8).
                // Starting from the first-step value (approximately 0.9),
                // the independently evaluated result is approximately 0.84226498.
                assertThat(parameter.tensor().toFloatArray()[0])
                        .isCloseTo(0.842265f, offset(3.0e-5f));
            } finally {
                firstGradient.release();
                secondGradient.release();
            }
        }
    }
}
