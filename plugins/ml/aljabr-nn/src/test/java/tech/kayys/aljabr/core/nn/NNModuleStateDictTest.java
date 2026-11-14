package tech.kayys.aljabr.core.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class NNModuleStateDictTest {

    @Test
    void stateDictExposesNamedParametersAndLoadsMatchingValues() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Tensor sourceWeight = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f }, backend, 2, 2);
            Tensor sourceBias = TensorFactory.of(new float[] { 5f, 6f }, backend, 2);
            try {
                layer.loadStateDict(Map.of("weight", sourceWeight, "bias", sourceBias));

                assertThat(layer.stateDict()).containsKeys("weight", "bias");
                assertThat(layer.stateDict().get("weight").toFloatArray())
                        .containsExactly(1f, 2f, 3f, 4f);
                assertThat(layer.stateDict().get("bias").toFloatArray())
                        .containsExactly(5f, 6f);
                assertThatThrownBy(() -> layer.stateDict().put("unexpected", sourceBias))
                        .isInstanceOf(UnsupportedOperationException.class);
            } finally {
                sourceWeight.release();
                sourceBias.release();
            }
        }
    }

    @Test
    void zeroGradReplacesAndReleasesThePreviousParameterGradient() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Tensor parameter = layer.stateDict().get("weight");
            Tensor previousGradient = TensorFactory.of(
                    new float[] { 1f, 2f, 3f, 4f }, backend, 2, 2);
            parameter.setGrad(previousGradient);

            layer.zeroGrad();

            assertThat(parameter.grad().toFloatArray()).containsExactly(0f, 0f, 0f, 0f);
            assertThatThrownBy(previousGradient::toFloatArray)
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("released");
        } finally {
            backend.close();
        }
    }

    @Test
    void strictLoadRejectsMissingAndUnexpectedKeysBeforeMutation() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            float[] before = layer.stateDict().get("weight").toFloatArray();
            Tensor replacement = TensorFactory.of(new float[] { 9f, 9f, 9f, 9f }, backend, 2, 2);
            try {
                assertThatThrownBy(() -> layer.loadStateDict(Map.of("weight", replacement)))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("missing=[bias]");
                assertThat(layer.stateDict().get("weight").toFloatArray()).containsExactly(before);
            } finally {
                replacement.release();
            }

            Tensor weight = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f }, backend, 2, 2);
            Tensor bias = TensorFactory.of(new float[] { 1f, 2f }, backend, 2);
            try {
                Map<String, Tensor> withUnexpected = new LinkedHashMap<>();
                withUnexpected.put("weight", weight);
                withUnexpected.put("bias", bias);
                withUnexpected.put("extra", weight);
                assertThatThrownBy(() -> layer.loadStateDict(withUnexpected))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("unexpected=[extra]");
            } finally {
                weight.release();
                bias.release();
            }
        }
    }

    @Test
    void loadRejectsDifferentShapeEvenWhenElementCountMatches() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Tensor wrongShape = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f }, backend, 4);
            Tensor bias = TensorFactory.of(new float[] { 1f, 2f }, backend, 2);
            try {
                assertThatThrownBy(() -> layer.loadStateDict(Map.of("weight", wrongShape, "bias", bias)))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("Shape mismatch for state weight");
            } finally {
                wrongShape.release();
                bias.release();
            }
        }
    }

    @Test
    void nonStrictLoadAllowsPartialState() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 2, 2, true)) {
            Tensor replacement = TensorFactory.of(new float[] { 7f, 8f, 9f, 10f }, backend, 2, 2);
            try {
                layer.loadStateDict(Map.of("weight", replacement, "future.extra", replacement), false);
                assertThat(layer.stateDict().get("weight").toFloatArray())
                        .containsExactly(7f, 8f, 9f, 10f);
            } finally {
                replacement.release();
            }
        }
    }
}
