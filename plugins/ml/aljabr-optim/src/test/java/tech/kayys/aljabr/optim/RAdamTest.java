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
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, -0.1f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 1.0f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, Float.NaN, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, 1.0f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, 0.999f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new RAdam(0.1f, 0.9f, 0.999f, Float.POSITIVE_INFINITY));
    }

    @Test
    void usesUnrectifiedFirstMomentDuringEarlySteps() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor secondGradient = TensorFactory.zeros(backend, 1, 1);
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
                    assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(0.9f, offset(2.0e-6f));

                    optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                    // rho_inf=3, so rho_1=1 and rho_2=5/3: neither step is rectified.
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo(0.8666667f, offset(3.0e-6f));
                } finally {
                    firstGradient.release();
                    secondGradient.release();
                }
            }
        }
    }

    @Test
    void matchesIndependentScalarReferenceAcrossRectificationBoundary() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.25f }, backend, 1, 1);
            float[] gradients = { 1.0f, 0.5f, -0.25f, 0.75f, 0.1f, -0.4f };
            try (Linear layer = new Linear(initial, null);
                 RAdam optimizer = new RAdam(0.03f, 0.9f, 0.9f, 1.0e-8f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                double expected = 1.25;
                double first = 0.0;
                double second = 0.0;
                double rhoInfinity = 2.0 / (1.0 - 0.9) - 1.0;
                for (int index = 0; index < gradients.length; index++) {
                    double gradient = gradients[index];
                    int step = index + 1;
                    first = 0.9 * first + 0.1 * gradient;
                    second = 0.9 * second + 0.1 * gradient * gradient;
                    double firstHat = first / (1.0 - Math.pow(0.9, step));
                    double beta2Power = Math.pow(0.9, step);
                    double rho = rhoInfinity - 2.0 * step * beta2Power / (1.0 - beta2Power);
                    double update;
                    if (rho > 4.0) {
                        double rectification = Math.sqrt(
                                ((rho - 4.0) * (rho - 2.0) * rhoInfinity)
                                        / ((rhoInfinity - 4.0) * (rhoInfinity - 2.0) * rho));
                        double secondHat = second / (1.0 - beta2Power);
                        update = 0.03 * rectification * firstHat / (Math.sqrt(secondHat) + 1.0e-8);
                    } else {
                        update = 0.03 * firstHat;
                    }
                    expected -= update;

                    Tensor gradientTensor = TensorFactory.of(new float[] {(float) gradient}, backend, 1, 1);
                    try {
                        optimizer.step(Map.of("weight", parameter), Map.of("weight", gradientTensor));
                    } finally {
                        gradientTensor.release();
                    }
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo((float) expected, offset(3.0e-6f));
                }
            }
        }
    }

    @Test
    void parameterReplacementRestartsMomentsAndRectificationSchedule() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor replacement = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor referenceInitial = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.ones(backend, 1, 1);
            Tensor nextGradient = TensorFactory.full(-0.5f, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Linear referenceLayer = new Linear(referenceInitial, null);
                 RAdam optimizer = new RAdam(0.1f);
                 RAdam reference = new RAdam(0.1f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                Parameter referenceParameter = referenceLayer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                    parameter.updateTensor(replacement);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", nextGradient));
                    reference.step(Map.of("weight", referenceParameter), Map.of("weight", nextGradient));
                    assertThat(parameter.tensor().toFloatArray())
                            .containsExactly(referenceParameter.tensor().toFloatArray());
                } finally {
                    firstGradient.release();
                    nextGradient.release();
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
    void resetRestartsRectificationSchedule() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 1, 1, false);
             RAdam optimizer = new RAdam(0.01f, 0.9f, 0.9f, 1.0e-8f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            Tensor gradient = TensorFactory.ones(backend, 1, 1);
            try {
                for (int i = 0; i < 6; i++) {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                }
                optimizer.reset();
                Tensor current = TensorFactory.of(parameter.tensor().toFloatArray(), backend, 1, 1);
                try (Linear referenceLayer = new Linear(current, null);
                     RAdam fresh = new RAdam(0.01f, 0.9f, 0.9f, 1.0e-8f)) {
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
