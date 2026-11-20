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

class RAdamTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new RAdam(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, -0.1f, 0.99f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 1.0f, 0.99f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, Float.NaN, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, 1.0f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, 0.99f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, 0.99f, Float.POSITIVE_INFINITY));
    }

    @Test
    void usesBiasCorrectedFirstMomentBeforeVarianceRectificationIsAvailable() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
            Tensor secondGradient = TensorFactory.of(new float[] { 0.25f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 RAdam optimizer = new RAdam(0.1f, 0.5f, 0.5f, 1.0e-8f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                    assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(0.95f, offset(1.0e-6f));

                    optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo(0.95f - (1.0f / 30.0f), offset(1.0e-6f));
                    assertThat(optimizer.learningRate()).isEqualTo(0.1f);
                    assertThat(optimizer.beta1()).isEqualTo(0.5f);
                    assertThat(optimizer.beta2()).isEqualTo(0.5f);
                } finally {
                    firstGradient.release();
                    secondGradient.release();
                }
            }
        }
    }

    @Test
    void matchesScalarReferenceAcrossVarianceRectificationBoundary() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.5f }, backend, 1, 1);
            float[] gradients = { 0.25f, -0.5f, 0.75f, 0.1f, -0.2f, 0.4f, -0.3f };
            try (Linear layer = new Linear(initial, null);
                 RAdam optimizer = new RAdam(0.01f, 0.9f, 0.9f, 1.0e-8f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                double expected = 1.5;
                double firstMoment = 0.0;
                double secondMoment = 0.0;
                try {
                    for (int i = 0; i < gradients.length; i++) {
                        float gradientValue = gradients[i];
                        Tensor gradient = TensorFactory.of(new float[] { gradientValue }, backend, 1, 1);
                        try {
                            optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                        } finally {
                            gradient.release();
                        }

                        int step = i + 1;
                        firstMoment = 0.9 * firstMoment + 0.1 * gradientValue;
                        secondMoment = 0.9 * secondMoment + 0.1 * gradientValue * gradientValue;
                        double firstHat = firstMoment / (1.0 - Math.pow(0.9, step));
                        double rhoInfinity = 2.0 / (1.0 - 0.9) - 1.0;
                        double beta2Power = Math.pow(0.9, step);
                        double rho = rhoInfinity - 2.0 * step * beta2Power / (1.0 - beta2Power);
                        double update;
                        if (rho > 4.0) {
                            double rectification = Math.sqrt(
                                    ((rho - 4.0) * (rho - 2.0) * rhoInfinity)
                                            / ((rhoInfinity - 4.0) * (rhoInfinity - 2.0) * rho));
                            double secondHat = secondMoment / (1.0 - beta2Power);
                            update = 0.01 * rectification * firstHat / (Math.sqrt(secondHat) + 1.0e-8);
                        } else {
                            update = 0.01 * firstHat;
                        }
                        expected -= update;
                        assertThat(parameter.tensor().toFloatArray()[0])
                                .as("step %s (rho=%.5f)", step, rho)
                                .isCloseTo((float) expected, offset(2.0e-6f));
                    }
                } finally {
                    // The layer owns its parameter tensor; optimizer state is
                    // released by close().
                }
            }
        }
    }

    @Test
    void parameterReplacementRestartsMomentsAndRectificationSchedule() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor replacement = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 RAdam optimizer = new RAdam(0.1f, 0.9f, 0.9f, 1.0e-8f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    for (int i = 0; i < 6; i++) {
                        optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    }
                    parameter.updateTensor(replacement);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo(1.95f, offset(1.0e-6f));
                } finally {
                    gradient.release();
                }
            }
        }
    }

    @Test
    void validatesAllTrainableGradientsBeforePublishingUpdatesAndSkipsFrozenParameters() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             RAdam optimizer = new RAdam(0.1f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            bias.setTrainable(false);
            Tensor originalWeight = weight.tensor();
            Tensor originalBias = bias.tensor();
            Tensor validGradient = TensorFactory.ones(backend, 2, 2);
            Tensor invalidGradient = TensorFactory.zeros(backend, 3);
            try {
                assertThrows(IllegalArgumentException.class, () -> optimizer.step(
                        Map.of("weight", weight, "bias", bias),
                        Map.of("weight", validGradient, "bias", invalidGradient)));
                assertThat(weight.tensor()).isSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);

                optimizer.step(Map.of("weight", weight, "bias", bias), Map.of("weight", validGradient));
                assertThat(weight.tensor()).isNotSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);
                optimizer.reset();
                Tensor beforeReset = weight.tensor();
                optimizer.step(Map.of("weight", weight), Map.of("weight", validGradient));
                assertThat(weight.tensor()).isNotSameAs(beforeReset);
            } finally {
                validGradient.release();
                invalidGradient.release();
            }
        }
    }
}
