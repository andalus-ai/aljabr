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

class RMSPropTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new RMSProp(-0.1f, 0.9f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RMSProp(Float.NaN, 0.9f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RMSProp(0.1f, -0.1f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RMSProp(0.1f, 1.0f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RMSProp(0.1f, Float.NaN, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new RMSProp(0.1f, 0.9f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new RMSProp(0.1f, 0.9f, Float.POSITIVE_INFINITY));
    }

    @Test
    void matchesIndependentTwoStepScalarReference() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.5f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] { 0.25f }, backend, 1, 1);
            Tensor secondGradient = TensorFactory.of(new float[] { -0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 RMSProp optimizer = new RMSProp(0.1f, 0.5f, 0.01f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    double squareAverage1 = 0.5 * 0.25 * 0.25;
                    double expected1 = 1.5 - 0.1 * 0.25 / (Math.sqrt(squareAverage1) + 0.01);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                    assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo((float) expected1, offset(1.0e-6f));

                    double squareAverage2 = 0.5 * squareAverage1 + 0.5 * 0.5 * 0.5;
                    double expected2 = expected1 - 0.1 * -0.5 / (Math.sqrt(squareAverage2) + 0.01);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                    assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo((float) expected2, offset(1.0e-6f));
                } finally {
                    firstGradient.release();
                    secondGradient.release();
                }
            }
        }
    }

    @Test
    void parameterReplacementStartsWithFreshSquareAverage() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor replacement = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 RMSProp optimizer = new RMSProp(0.1f, 0.9f, 0.01f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    parameter.updateTensor(replacement);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));

                    double freshSquareAverage = (1.0 - 0.9) * 0.5 * 0.5;
                    double expected = 2.0 - 0.1 * 0.5 / (Math.sqrt(freshSquareAverage) + 0.01);
                    assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo((float) expected, offset(1.0e-6f));
                } finally {
                    gradient.release();
                }
            }
        }
    }

    @Test
    void validatesAllGradientsBeforeUpdatingAnyParameter() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             RMSProp optimizer = new RMSProp(0.1f, 0.9f, 1.0e-8f)) {
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
    void missingAndFrozenParametersDoNotRequireGradients() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             RMSProp optimizer = new RMSProp(0.1f, 0.9f, 1.0e-8f)) {
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
            } finally {
                gradient.release();
            }
        }
    }

    @Test
    void resetClearsSquareAverageWithoutChangingHyperparameters() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 RMSProp optimizer = new RMSProp(0.1f, 0.9f, 0.01f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    optimizer.reset();
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));

                    double freshSquareAverage = (1.0 - 0.9) * 0.5 * 0.5;
                    double expected = (1.0 - 0.1 * 0.5 / (Math.sqrt(freshSquareAverage) + 0.01));
                    assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo((float) expected, offset(1.0e-6f));
                    assertThat(optimizer.learningRate()).isEqualTo(0.1f);
                    assertThat(optimizer.alpha()).isEqualTo(0.9f);
                    assertThat(optimizer.epsilon()).isEqualTo(0.01f);
                } finally {
                    gradient.release();
                }
            }
        }
    }
}
