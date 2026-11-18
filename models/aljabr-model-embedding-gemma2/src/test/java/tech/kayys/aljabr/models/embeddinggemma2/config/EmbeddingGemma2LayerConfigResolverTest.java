package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class EmbeddingGemma2LayerConfigResolverTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void resolvesBaseSlidingAttentionGeometry() throws Exception {
        EmbeddingGemma2LayerConfig layer = resolve(0);
        assertEquals(EmbeddingGemma2LayerConfig.AttentionType.SLIDING, layer.attentionType());
        assertEquals(4, layer.numAttentionHeads());
        assertEquals(2, layer.numKeyValueHeads());
        assertEquals(256, layer.headDim());
        assertEquals(1024, layer.queryProjectionSize());
        assertEquals(512, layer.keyValueProjectionSize());
        assertEquals(2, layer.queriesPerKeyValueHead());
        assertEquals(10000.0d, layer.ropeTheta());
        assertEquals("default", layer.ropeType());
        assertEquals(512, layer.slidingWindow());
    }

    @Test
    void resolvesFullAttentionPerLayerOverridesAndProportionalRope() throws Exception {
        JsonNode json = validConfig();
        ((com.fasterxml.jackson.databind.node.ObjectNode) json.path("text_config")
                .path("rope_parameters").path("full_attention"))
                .put("rope_type", "proportional").put("partial_rotary_factor", 0.25d);
        EmbeddingGemma2Config config = EmbeddingGemma2Config.fromJson(json);

        EmbeddingGemma2LayerConfig layer =
                EmbeddingGemma2LayerConfigResolver.resolve(config, 5);
        assertEquals(EmbeddingGemma2LayerConfig.AttentionType.FULL, layer.attentionType());
        assertEquals(4, layer.numAttentionHeads());
        assertEquals(1, layer.numKeyValueHeads());
        assertEquals(512, layer.headDim());
        assertEquals(2048, layer.queryProjectionSize());
        assertEquals(512, layer.keyValueProjectionSize());
        assertEquals(4, layer.queriesPerKeyValueHead());
        assertEquals(1000000.0d, layer.ropeTheta());
        assertEquals("proportional", layer.ropeType());
        assertEquals(0.25d, layer.partialRotaryFactor());
        assertNull(layer.slidingWindow());
    }

    @Test
    void rejectsUnknownLayerTypesAndIncompleteLayerLayouts() throws Exception {
        JsonNode json = validConfig();
        ((com.fasterxml.jackson.databind.node.ArrayNode) json.path("text_config").path("layer_types"))
                .set(0, mapper.getNodeFactory().textNode("causal_attention"));
        EmbeddingGemma2Config config = EmbeddingGemma2Config.fromJson(json);
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2LayerConfigResolver.resolve(config, 0));

        JsonNode incomplete = validConfig();
        ((com.fasterxml.jackson.databind.node.ObjectNode) incomplete.path("text_config"))
                .remove("layer_types");
        EmbeddingGemma2Config noLayerTypes = EmbeddingGemma2Config.fromJson(incomplete);
        assertThrows(IllegalStateException.class,
                () -> EmbeddingGemma2LayerConfigResolver.resolve(noLayerTypes, 0));
    }

    @Test
    void rejectsOutOfRangeLayerIndex() throws Exception {
        EmbeddingGemma2Config config = EmbeddingGemma2Config.fromJson(validConfig());
        assertThrows(IndexOutOfBoundsException.class,
                () -> EmbeddingGemma2LayerConfigResolver.resolve(config, 24));
    }

    private EmbeddingGemma2LayerConfig resolve(int layerIndex) throws Exception {
        return EmbeddingGemma2LayerConfigResolver.resolve(
                EmbeddingGemma2Config.fromJson(validConfig()), layerIndex);
    }

    private JsonNode validConfig() throws Exception {
        StringBuilder types = new StringBuilder("[");
        for (int i = 0; i < 24; i++) {
            if (i > 0) types.append(',');
            types.append(i == 5 || i == 11 || i == 17 || i == 23
                    ? "\"full_attention\"" : "\"sliding_attention\"");
        }
        types.append(']');
        String json = """
                {
                  "model_type":"embedding_gemma2",
                  "text_config":{
                    "model_type":"embedding_gemma2_text",
                    "hidden_size":512,
                    "intermediate_size":2048,
                    "num_hidden_layers":24,
                    "num_attention_heads":4,
                    "num_key_value_heads":2,
                    "head_dim":256,
                    "embedding_dim":768,
                    "vocab_size":262144,
                    "max_position_embeddings":262144,
                    "dtype":"bfloat16",
                    "rms_norm_eps":0.000001,
                    "rope_parameters":{
                      "full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
                      "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}
                    },
                    "sliding_window":512,
                    "layer_types":__LAYER_TYPES__,
                    "per_layer_config":{
                      "05":{"head_dim":512,"num_key_value_heads":1},
                      "11":{"head_dim":512,"num_key_value_heads":1},
                      "17":{"head_dim":512,"num_key_value_heads":1},
                      "23":{"head_dim":512,"num_key_value_heads":1}
                    }
                  }
                }
                """.replace("__LAYER_TYPES__", types.toString());
        return mapper.readTree(json);
    }
}
