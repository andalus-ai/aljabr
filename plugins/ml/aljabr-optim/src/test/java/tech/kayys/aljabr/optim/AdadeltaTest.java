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

class AdadeltaTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new Adadelta(-0.1f, 1.0e-6f));
        assertThrows(IllegalArgumentException.class, () -> new Adadelta(1.0f, 1.0e-6f));
        assertThrows(IllegalArgumentException.class, () -> new Adadelta(Float.NaN, 1.0e-6f));
        assertThrows(IllegalArgumentException.class, () -> new Adadelta(0.9f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new Adadelta(0.9f, Float.POSITIVE_INFINITY));
    }

    @Test
    void matchesIndependentTwoStepScalarReference() {
        double rho = 0.5;
        double epsilon = 0.01;
        double expected = 1.5;
        double gradientSquares = 0.0;
        double updateSquares = 0.0;
        float[] gradients = {0.25f, -0.5f};

        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] {1.5f}, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] {gradients[0]}, backend, 1, 1);
            Tensor secondGradient = TensorFactory.of(new float[] {gradients[1]}, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Adadelta optimizer = new Adadelta((float) rho, (float) epsilon)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    Tensor[] supplied = {firstGradient, secondGradient};
                    for (int index = 0; index < gradients.length; index++) {
                        double gradient = gradients[index];
                        gradientSquares = rho * gradientSquares + (1.0 - rho) * gradient * gradient;
                        double delta = -Math.sqrt(updateSquares + epsilon)
                                / Math.sqrt(gradientSquares + epsilon) * gradient;
                        updateSquares = rho * updateSquares + (1.0 - rho) * delta * delta;
                        expected += delta;

                        optimizer.step(Map.of("weight", parameter), Map.of("weight", supplied[index]));
                        assertThat(parameter.tensor().toFloatArray()[0])
                                .isCloseTo((float) expected, offset(1.0e-6f));
                    }
                } finally {
                    firstGradient.release();
                    secondGradient.release();
                }
            }
        }
    }

    @Test
    void replacementStartsWithFreshAccumulatorState() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] {1.0f}, backend, 1, 1);
            Tensor replacement = TensorFactory.of(new float[] {2.0f}, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] {0.5f}, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Adadelta optimizer = new Adadelta(0.9f, 0.01f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    parameter.updateTensor(replacement);
                    double freshGradientSquares = 0.1 * 0.25;
                    double delta = -Math.sqrt(0.01) / Math.sqrt(freshGradientSquares + 0.01) * 0.5;
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo((float) (2.0 + delta), offset(1.0e-6f));
                } finally {
                    gradient.release();
                }
            }
        }
    }

    @Test
    void validatesAllGradientsBeforePublishingAnyUpdate() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Adadelta optimizer = new Adadelta(0.9f, 1.0e-6f)) {
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
    void missingAndFrozenParametersAreSkipped() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Adadelta optimizer = new Adadelta(0.9f, 1.0e-6f)) {
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
    void resetRestartsBothAccumulatorsAndPreservesHyperparameters() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] {1.0f}, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] {0.5f}, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Adadelta optimizer = new Adadelta(0.9f, 0.01f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    optimizer.reset();

                    double gradientSquares = 0.1 * 0.25;
                    double delta = -Math.sqrt(0.01) / Math.sqrt(gradientSquares + 0.01) * 0.5;
                    double expected = parameter.tensor().toFloatArray()[0] + delta;
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo((float) expected, offset(1.0e-6f));
                    assertThat(optimizer.rho()).isEqualTo(0.9f);
                    assertThat(optimizer.epsilon()).isEqualTo(0.01f);
                } finally {
                    gradient.release();
                }
            }
        }
    }
}
