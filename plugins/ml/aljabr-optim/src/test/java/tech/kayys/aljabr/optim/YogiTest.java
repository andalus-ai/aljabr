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

class YogiTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new Yogi(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new Yogi(Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new Yogi(0.1f, -0.1f, 0.99f, 1.0e-3f));
        assertThrows(IllegalArgumentException.class,
                () -> new Yogi(0.1f, 1.0f, 0.99f, 1.0e-3f));
        assertThrows(IllegalArgumentException.class,
                () -> new Yogi(0.1f, 0.9f, Float.NaN, 1.0e-3f));
        assertThrows(IllegalArgumentException.class,
                () -> new Yogi(0.1f, 0.9f, 1.0f, 1.0e-3f));
        assertThrows(IllegalArgumentException.class,
                () -> new Yogi(0.1f, 0.9f, 0.99f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new Yogi(0.1f, 0.9f, 0.99f, Float.POSITIVE_INFINITY));
    }

    @Test
    void matchesIndependentReferenceAcrossMultipleSteps() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            float[] gradients = { 1.0f, 0.0f, 0.5f, -0.2f, 0.1f };
            try (Linear layer = new Linear(initial, null);
                 Yogi optimizer = new Yogi(0.1f, 0.5f, 0.5f, 1.0e-3f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                double expected = 1.0;
                double firstMoment = 0.0;
                double secondMoment = 0.0;
                for (int i = 0; i < gradients.length; i++) {
                    float gradientValue = gradients[i];
                    Tensor gradient = TensorFactory.of(
                            new float[] { gradientValue }, backend, 1, 1);
                    try {
                        optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    } finally {
                        gradient.release();
                    }

                    int step = i + 1;
                    firstMoment = 0.5 * firstMoment + 0.5 * gradientValue;
                    double squaredGradient = gradientValue * gradientValue;
                    double difference = secondMoment - squaredGradient;
                    secondMoment -= 0.5 * squaredGradient * Math.signum(difference);
                    double firstHat = firstMoment / (1.0 - Math.pow(0.5, step));
                    double secondHat = secondMoment / (1.0 - Math.pow(0.5, step));
                    expected -= 0.1 * firstHat / (Math.sqrt(secondHat) + 1.0e-3);
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .as("step %s", step)
                            .isCloseTo((float) expected, offset(2.0e-6f));
                }
                assertThat(optimizer.learningRate()).isEqualTo(0.1f);
                assertThat(optimizer.beta1()).isEqualTo(0.5f);
                assertThat(optimizer.beta2()).isEqualTo(0.5f);
                assertThat(optimizer.epsilon()).isEqualTo(1.0e-3f);
            }
        }
    }

    @Test
    void replacementRestartsMomentsAndStepCounter() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor replacement = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Linear referenceLayer = new Linear(
                         TensorFactory.of(new float[] { 2.0f }, backend, 1, 1), null);
                 Yogi optimizer = new Yogi(0.1f, 0.5f, 0.5f, 1.0e-3f);
                 Yogi fresh = new Yogi(0.1f, 0.5f, 0.5f, 1.0e-3f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                Parameter reference = referenceLayer.namedParameterObjects().get("weight");
                try {
                    for (int i = 0; i < 3; i++) {
                        optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    }
                    parameter.updateTensor(replacement);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    fresh.step(Map.of("weight", reference), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo(reference.tensor().toFloatArray()[0], offset(1.0e-6f));
                } finally {
                    gradient.release();
                }
            }
        }
    }

    @Test
    void validatesGradientsBeforePublishingAndSkipsFrozenParameters() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Yogi optimizer = new Yogi(0.1f)) {
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
                optimizer.step(Map.of("weight", weight, "bias", bias),
                        Map.of("weight", valid, "bias", invalid));
                assertThat(weight.tensor()).isNotSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);

                optimizer.reset();
                Tensor beforeReset = weight.tensor();
                optimizer.step(Map.of("weight", weight), Map.of("weight", valid));
                assertThat(weight.tensor()).isNotSameAs(beforeReset);
            } finally {
                valid.release();
                invalid.release();
            }
        }
    }
}
