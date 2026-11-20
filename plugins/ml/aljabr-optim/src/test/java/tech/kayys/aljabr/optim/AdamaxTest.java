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

class AdamaxTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new Adamax(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new Adamax(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new Adamax(0.1f, -0.1f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Adamax(0.1f, 1.0f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Adamax(0.1f, 0.9f, Float.NaN, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Adamax(0.1f, 0.9f, 1.0f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Adamax(0.1f, 0.9f, 0.999f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new Adamax(0.1f, 0.9f, 0.999f, Float.POSITIVE_INFINITY));
    }

    @Test
    void matchesIndependentTwoStepInfinityNormReference() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor secondGradient = TensorFactory.zeros(backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Adamax optimizer = new Adamax(0.1f, 0.5f, 0.5f, 1.0e-8f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo(0.9f, offset(2.0e-6f));

                    optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                    // m=(0.5,0.25), u=(1.0,0.5); mHat_2=1/3.
                    // The second step is 0.1 * (1/3) / 0.5 = 1/15.
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo(0.8333333f, offset(3.0e-6f));
                    assertThat(optimizer.learningRate()).isEqualTo(0.1f);
                    assertThat(optimizer.beta1()).isEqualTo(0.5f);
                    assertThat(optimizer.beta2()).isEqualTo(0.5f);
                    assertThat(optimizer.epsilon()).isEqualTo(1.0e-8f);
                } finally {
                    firstGradient.release();
                    secondGradient.release();
                }
            }
        }
    }

    @Test
    void replacementRestartsMomentsAndBiasCorrection() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor replacement = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor referenceInitial = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor secondGradient = TensorFactory.of(new float[] { -0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Linear referenceLayer = new Linear(referenceInitial, null);
                 Adamax optimizer = new Adamax(0.1f, 0.5f, 0.5f, 1.0e-8f);
                 Adamax reference = new Adamax(0.1f, 0.5f, 0.5f, 1.0e-8f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                Parameter referenceParameter = referenceLayer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                    parameter.updateTensor(replacement);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                    reference.step(Map.of("weight", referenceParameter), Map.of("weight", secondGradient));
                    assertThat(parameter.tensor().toFloatArray())
                            .containsExactly(referenceParameter.tensor().toFloatArray());
                } finally {
                    firstGradient.release();
                    secondGradient.release();
                }
            }
        }
    }

    @Test
    void validatesAllGradientsBeforePublishingAnyParameterAndSkipsFrozenValues() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Adamax optimizer = new Adamax(0.1f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            Tensor originalWeight = weight.tensor();
            Tensor originalBias = bias.tensor();
            Tensor valid = TensorFactory.ones(backend, 2, 2);
            Tensor invalid = TensorFactory.zeros(backend, 3);
            try {
                assertThrows(IllegalArgumentException.class, () -> optimizer.step(
                        Map.of("weight", weight, "bias", bias),
                        Map.of("weight", valid, "bias", invalid)));
                assertThat(weight.tensor()).isSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);

                bias.setTrainable(false);
                optimizer.step(Map.of("weight", weight, "bias", bias), Map.of("weight", valid, "bias", invalid));
                assertThat(weight.tensor()).isNotSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);
            } finally {
                valid.release();
                invalid.release();
            }
        }
    }

    @Test
    void resetReleasesStateAndRestartsTheStepSchedule() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, false);
             Adamax optimizer = new Adamax(0.01f, 0.5f, 0.5f, 1.0e-8f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                optimizer.reset();
                Tensor currentValue = TensorFactory.of(parameter.tensor().toFloatArray(), backend, 2, 2);
                try (Linear referenceLayer = new Linear(currentValue, null);
                     Adamax fresh = new Adamax(0.01f, 0.5f, 0.5f, 1.0e-8f)) {
                    Parameter referenceParameter = referenceLayer.namedParameterObjects().get("weight");
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    fresh.step(Map.of("weight", referenceParameter), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray())
                            .containsExactly(referenceParameter.tensor().toFloatArray());
                }
            } finally {
                gradient.release();
            }
        }
    }
}
