package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2EncoderLayer;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;

class EmbeddingGemma2TextBackboneTest {

    @Test
    void rejectsLayerCountThatDoesNotMatchConfiguration() {
        EmbeddingGemma2Config config = minimalConfig();
        EmbeddingGemma2TextBackbone.Weights weights = new EmbeddingGemma2TextBackbone.Weights(
                new float[][]{{1f, 1f}}, new float[][]{{1f, 1f}},
                new float[]{1f}, new float[]{1f}, new float[][]{{1f, 0f}});
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2TextBackbone(config, List.of(), weights));
    }

    @Test
    void rejectsMalformedCheckpointEmbeddingMatrix() {
        EmbeddingGemma2Config config = minimalConfig();
        EmbeddingGemma2TextBackbone.Weights weights = new EmbeddingGemma2TextBackbone.Weights(
                new float[][]{{1f}}, new float[][]{{1f, 1f}},
                new float[]{1f}, new float[]{1f}, new float[][]{{1f, 0f}});
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2TextBackbone(config, List.of(dummyLayer()), weights));
    }

    private static EmbeddingGemma2EncoderLayer dummyLayer() {
        // A null layer is deliberately never executed by these constructor validation tests.
        return null;
    }

    private static EmbeddingGemma2Config minimalConfig() {
        return new EmbeddingGemma2Config(
                EmbeddingGemma2Config.MODEL_TYPE,
                EmbeddingGemma2Config.TEXT_MODEL_TYPE,
                2, 2, 1, 1, 1, 2, 1,
                EmbeddingGemma2Config.DEFAULT_HIDDEN_ACTIVATION,
                1, 2, 8, 8, "float32", 1e-6,
                null, List.of("full_attention"), java.util.Map.of(),
                com.fasterxml.jackson.databind.node.JsonNodeFactory.instance.objectNode()
                        .putObject("full_attention").put("rope_theta", 10000.0)
                        .put("rope_type", "default").parent().set("sliding_attention",
                                com.fasterxml.jackson.databind.node.JsonNodeFactory.instance.objectNode()
                                        .put("rope_theta", 10000.0).put("rope_type", "default")));
    }
}
