package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertThrows;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;

class EmbeddingGemma2TextBackboneTest {
    @Test
    void rejectsLayerCountThatDoesNotMatchConfiguration() {
        EmbeddingGemma2Config config = minimalConfig();
        var weights = new EmbeddingGemma2TextBackbone.Weights(
                new float[][]{{1f, 1f}, {1f, 1f}},
                new float[][]{{1f, 1f}}, new float[]{1f}, new float[]{1f},
                new float[][]{{1f, 0f}});
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2TextBackbone(config, List.of(), weights));
    }

    private static EmbeddingGemma2Config minimalConfig() {
        ObjectNode rope = JsonNodeFactory.instance.objectNode();
        rope.putObject("full_attention").put("rope_theta", 10000.0).put("rope_type", "default");
        rope.putObject("sliding_attention").put("rope_theta", 10000.0).put("rope_type", "default");
        return new EmbeddingGemma2Config(EmbeddingGemma2Config.MODEL_TYPE,
                EmbeddingGemma2Config.TEXT_MODEL_TYPE, 2, 2, 1, 1, 1, 2, 1,
                EmbeddingGemma2Config.DEFAULT_HIDDEN_ACTIVATION, 1, 2, 8, 8,
                "float32", 1e-6, null, List.of("full_attention"), Map.of(), rope);
    }
}
