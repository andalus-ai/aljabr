package tech.kayys.aljabr.optim;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class LionTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new Lion(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new Lion(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new Lion(0.1f, -0.1f, 0.99f));
        assertThrows(IllegalArgumentException.class, () -> new Lion(0.1f, 1.0f, 0.99f));
        assertThrows(IllegalArgumentException.class, () -> new Lion(0.1f, 0.9f, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new Lion(0.1f, 0.9f, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> new Lion(0.1f, 0.9f, 0.99f, -0.1f));
        assertThrows(IllegalArgumentException.class, () -> new Lion(0.1f, 0.9f, 0.99f, Float.POSITIVE_INFINITY));
    }

    @Test
    void matchesIndependentTwoStepScalarReferenceWithDecoupledWeightDecay() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.5f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] { 0.25f }, backend, 1, 1);
            Tensor secondGradient = TensorFactory.of(new float[] { -0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Lion optimizer = new Lion(0.1f, 0.9f, 0.99f, 0.01f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    double firstMomentum = (1.0 - 0.99) * 0.25;
                    double firstDirectionInput = 0.9 * firstMomentum + 0.1 * 0.25;
                    double expectedFirst = 1.5 * (1.0 - 0.1 * 0.01)
                            - 0.1 * Math.signum(firstDirectionInput);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo((float) expectedFirst, offset(1.0e-6f));

                    double secondMomentum = 0.99 * firstMomentum + 0.01 * -0.5;
                    double secondDirectionInput = 0.9 * secondMomentum + 0.1 * -0.5;
                    double expectedSecond = expectedFirst * (1.0 - 0.1 * 0.01)
                            - 0.1 * Math.signum(secondDirectionInput);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo((float) expectedSecond, offset(1.0e-6f));
                    assertThat(optimizer.learningRate()).isEqualTo(0.1f);
                    assertThat(optimizer.beta1()).isEqualTo(0.9f);
                    assertThat(optimizer.beta2()).isEqualTo(0.99f);
                    assertThat(optimizer.weightDecay()).isEqualTo(0.01f);
                } finally {
                    firstGradient.release();
                    secondGradient.release();
                }
            }
        }
    }

    @Test
    void parameterReplacementRestartsMomentum() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor replacement = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Lion optimizer = new Lion(0.1f, 0.9f, 0.99f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    parameter.updateTensor(replacement);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));

                    double freshMomentum = (1.0 - 0.99) * 0.5;
                    double directionInput = 0.9 * freshMomentum + 0.1 * 0.5;
                    double expected = 2.0 - 0.1 * Math.signum(directionInput);
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo((float) expected, offset(1.0e-6f));
                } finally {
                    gradient.release();
                }
            }
        }
    }

    @Test
    void validatesAllGradientsBeforePublishingAnyParameter() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Lion optimizer = new Lion(0.1f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            Tensor originalWeight = weight.tensor();
            Tensor validWeightGradient = TensorFactory.ones(backend, 2, 2);
            Tensor invalidBiasGradient = TensorFactory.zeros(backend, 3);
            try {
                assertThrows(IllegalArgumentException.class, () -> optimizer.step(
                        Map.of("weight", weight, "bias", bias),
                        Map.of("weight", validWeightGradient, "bias", invalidBiasGradient)));
                assertThat(weight.tensor()).isSameAs(originalWeight);
            } finally {
                validWeightGradient.release();
                invalidBiasGradient.release();
            }
        }
    }

    @Test
    void missingAndFrozenParametersAreSkippedAndResetClearsMomentum() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Lion optimizer = new Lion(0.1f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            bias.setTrainable(false);
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            try {
                Tensor originalWeight = weight.tensor();
                Tensor originalBias = bias.tensor();
                optimizer.step(Map.of("weight", weight, "bias", bias), Map.of());
                assertThat(weight.tensor()).isSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);

                optimizer.step(Map.of("weight", weight, "bias", bias), Map.of("weight", gradient));
                assertThat(weight.tensor()).isNotSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);

                optimizer.reset();
                Tensor beforeResetStep = weight.tensor();
                optimizer.step(Map.of("weight", weight), Map.of("weight", gradient));
                assertThat(weight.tensor()).isNotSameAs(beforeResetStep);
            } finally {
                gradient.release();
            }
        }
    }
}
