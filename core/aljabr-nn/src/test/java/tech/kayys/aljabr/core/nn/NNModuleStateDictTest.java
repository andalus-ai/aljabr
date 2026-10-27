package tech.kayys.aljabr.core.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;

class NNModuleStateDictTest {

    @Test
    void stateDictExposesNamedParametersAndLoadsMatchingValues() {
        try (Linear layer = new Linear(2, 2, true)) {
            Tensor sourceWeight = Tensor.of(new float[] { 1f, 2f, 3f, 4f }, 2, 2);
            Tensor sourceBias = Tensor.of(new float[] { 5f, 6f }, 2);
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
    void strictLoadRejectsMissingAndUnexpectedKeysBeforeMutation() {
        try (Linear layer = new Linear(2, 2, true)) {
            float[] before = layer.stateDict().get("weight").toFloatArray();
            Tensor replacement = Tensor.of(new float[] { 9f, 9f, 9f, 9f }, 2, 2);
            try {
                assertThatThrownBy(() -> layer.loadStateDict(Map.of("weight", replacement)))
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessageContaining("missing=[bias]");
                assertThat(layer.stateDict().get("weight").toFloatArray()).containsExactly(before);
            } finally {
                replacement.release();
            }

            Tensor weight = Tensor.of(new float[] { 1f, 2f, 3f, 4f }, 2, 2);
            Tensor bias = Tensor.of(new float[] { 1f, 2f }, 2);
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
        try (Linear layer = new Linear(2, 2, true)) {
            Tensor wrongShape = Tensor.of(new float[] { 1f, 2f, 3f, 4f }, 4);
            Tensor bias = Tensor.of(new float[] { 1f, 2f }, 2);
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
        try (Linear layer = new Linear(2, 2, true)) {
            Tensor replacement = Tensor.of(new float[] { 7f, 8f, 9f, 10f }, 2, 2);
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
