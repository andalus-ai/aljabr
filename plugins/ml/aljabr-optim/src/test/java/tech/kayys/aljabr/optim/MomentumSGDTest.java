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

class MomentumSGDTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new MomentumSGD(-0.1f, 0.5f));
        assertThrows(IllegalArgumentException.class, () -> new MomentumSGD(Float.NaN, 0.5f));
        assertThrows(IllegalArgumentException.class, () -> new MomentumSGD(0.1f, -0.1f));
        assertThrows(IllegalArgumentException.class, () -> new MomentumSGD(0.1f, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> new MomentumSGD(0.1f, Float.NaN));
    }

    @Test
    void matchesIndependentTwoStepMomentumReference() {
        CpuBackend backend = new CpuBackend();
        Tensor initial = TensorFactory.of(new float[] { 1.5f }, backend, 1, 1);
        Tensor firstGradient = TensorFactory.of(new float[] { 0.25f }, backend, 1, 1);
        Tensor secondGradient = TensorFactory.of(new float[] { -0.5f }, backend, 1, 1);
        try (Linear layer = new Linear(initial, null);
             MomentumSGD optimizer = new MomentumSGD(0.1f, 0.5f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", firstGradient));
                // v1 = 0.25; p1 = 1.5 - (0.1 * 0.25) = 1.475.
                assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(1.475f, offset(1.0e-6f));

                optimizer.step(Map.of("weight", parameter), Map.of("weight", secondGradient));
                // v2 = (0.5 * 0.25) + (-0.5) = -0.375; p2 = 1.475 - (0.1 * -0.375).
                assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(1.5125f, offset(1.0e-6f));
            } finally {
                firstGradient.release();
                secondGradient.release();
            }
        }
    }

    @Test
    void validatesAllGradientsBeforeUpdatingAnyParameter() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true);
             MomentumSGD optimizer = new MomentumSGD(0.1f, 0.9f)) {
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
    void parameterReplacementStartsWithFreshMomentum() {
        CpuBackend backend = new CpuBackend();
        Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
        Tensor replacement = TensorFactory.of(new float[] { 2.0f }, backend, 1, 1);
        Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
        try (Linear layer = new Linear(initial, null);
             MomentumSGD optimizer = new MomentumSGD(0.1f, 0.9f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                parameter.updateTensor(replacement);
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                // Fresh velocity is 0.5, not 0.9 * 0.5 + 0.5.
                assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(1.95f, offset(1.0e-6f));
            } finally {
                gradient.release();
            }
        }
    }

    @Test
    void resetClearsMomentumWithoutChangingHyperparameters() {
        CpuBackend backend = new CpuBackend();
        Tensor initial = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1);
        Tensor gradient = TensorFactory.of(new float[] { 0.5f }, backend, 1, 1);
        try (Linear layer = new Linear(initial, null);
             MomentumSGD optimizer = new MomentumSGD(0.1f, 0.9f)) {
            Parameter parameter = layer.namedParameterObjects().get("weight");
            try {
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                optimizer.reset();
                optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                assertThat(parameter.tensor().toFloatArray()[0]).isCloseTo(0.9f, offset(1.0e-6f));
                assertThat(optimizer.learningRate()).isEqualTo(0.1f);
                assertThat(optimizer.momentum()).isEqualTo(0.9f);
            } finally {
                gradient.release();
            }
        }
    }
}
