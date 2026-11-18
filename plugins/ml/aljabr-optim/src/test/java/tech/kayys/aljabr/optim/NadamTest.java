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

class NadamTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new Nadam(-0.1f, 0.9f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Nadam(Float.NaN, 0.9f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Nadam(0.1f, -0.1f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Nadam(0.1f, 1.0f, 0.999f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Nadam(0.1f, 0.9f, Float.NaN, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Nadam(0.1f, 0.9f, 1.0f, 1.0e-8f));
        assertThrows(IllegalArgumentException.class, () -> new Nadam(0.1f, 0.9f, 0.999f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new Nadam(0.1f, 0.9f, 0.999f, Float.POSITIVE_INFINITY));
    }

    @Test
    void matchesIndependentTwoStepDozatReference() {
        float learningRate = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1.0e-8f;
        double expected = 2.0;
        double firstMoment = 0.0;
        double secondMoment = 0.0;
        float[] gradients = { 0.5f, -0.25f };

        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
            Tensor firstGradient = TensorFactory.of(new float[] { gradients[0] }, backend, 1, 1);
            Tensor secondGradient = TensorFactory.of(new float[] { gradients[1] }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Nadam optimizer = new Nadam(learningRate, beta1, beta2, epsilon)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    Tensor[] gradientTensors = { firstGradient, secondGradient };
                    for (int index = 0; index < gradients.length; index++) {
                        long step = index + 1L;
                        double gradient = gradients[index];
                        firstMoment = beta1 * firstMoment + (1.0 - beta1) * gradient;
                        secondMoment = beta2 * secondMoment + (1.0 - beta2) * gradient * gradient;
                        double correction1 = 1.0 - Math.pow(beta1, step);
                        double correction2 = 1.0 - Math.pow(beta2, step);
                        double lookahead = beta1 * firstMoment / correction1
                                + (1.0 - beta1) * gradient / correction1;
                        double variance = secondMoment / correction2;
                        expected -= learningRate * lookahead / (Math.sqrt(variance) + epsilon);

                        optimizer.step(Map.of("weight", parameter),
                                Map.of("weight", gradientTensors[index]));
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
    void resetRestartsBiasCorrectionAndPreservesHyperparameters() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
            Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
            try (Linear layer = new Linear(initial, null);
                 Nadam optimizer = new Nadam(0.001f, 0.9f, 0.999f, 1.0e-8f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    optimizer.reset();
                    double firstMoment = 0.1 * 0.5;
                    double secondMoment = 0.001 * 0.25;
                    double lookahead = 0.9 * firstMoment / 0.1 + 0.1 * 0.5 / 0.1;
                    double variance = secondMoment / 0.001;
                    double expected = parameter.tensor().toFloatArray()[0]
                            - 0.001 * lookahead / (Math.sqrt(variance) + 1.0e-8);
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));

                    assertThat(parameter.tensor().toFloatArray()[0])
                            .isCloseTo((float) expected, offset(1.0e-6f));
                    assertThat(optimizer.learningRate()).isEqualTo(0.001f);
                    assertThat(optimizer.beta1()).isEqualTo(0.9f);
                    assertThat(optimizer.beta2()).isEqualTo(0.999f);
                    assertThat(optimizer.epsilon()).isEqualTo(1.0e-8f);
                } finally {
                    gradient.release();
                }
            }
        }
    }
}
