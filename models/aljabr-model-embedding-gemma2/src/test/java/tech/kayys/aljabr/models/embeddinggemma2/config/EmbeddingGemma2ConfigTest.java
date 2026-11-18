package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingGemma2ConfigTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void parsesNestedTextConfigAndPreservesPerLayerOverrides() throws Exception {
        EmbeddingGemma2Config config = EmbeddingGemma2Config.fromJson(validConfig());

        assertEquals("embedding_gemma2", config.modelType());
        assertEquals("embedding_gemma2_text", config.textModelType());
        assertEquals(512, config.hiddenSize());
        assertEquals(2048, config.intermediateSize());
        assertEquals(24, config.numHiddenLayers());
        assertEquals(4, config.numAttentionHeads());
        assertEquals(2, config.numKeyValueHeads());
        assertEquals(262144, config.vocabSize());
        assertEquals(262144, config.maxPositionEmbeddings());
        assertEquals(8192, config.maxInputTokens());
        assertEquals(512, config.slidingWindow());
        assertEquals(4, config.perLayerConfig().size());
        assertEquals(512, config.perLayerOverride(5).path("head_dim").asInt());
        assertNotNull(config.perLayerOverride(23));
    }

    @Test
    void rejectsCausalGemmaConfigInsteadOfAccidentallyAcceptingIt() throws Exception {
        JsonNode json = mapper.readTree("""
                {"model_type":"gemma4","text_config":{"model_type":"gemma4_text"}}
                """);
        assertThrows(IllegalArgumentException.class, () -> EmbeddingGemma2Config.fromJson(json));
    }

    @Test
    void rejectsUnsupportedPrecision() throws Exception {
        JsonNode json = validConfig();
        ((ObjectNode) json.path("text_config")).put("dtype", "float16");
        assertThrows(IllegalArgumentException.class, () -> EmbeddingGemma2Config.fromJson(json));
    }

    @Test
    void rejectsInvalidLayerOverrideIndex() throws Exception {
        JsonNode json = validConfig();
        ((ObjectNode) json.path("text_config").path("per_layer_config"))
                .set("24", mapper.createObjectNode().put("head_dim", 256));
        assertThrows(IllegalArgumentException.class, () -> EmbeddingGemma2Config.fromJson(json));
    }

    @Test
    void rejectsLayerTypeCountMismatch() throws Exception {
        JsonNode json = validConfig();
        ((ObjectNode) json.path("text_config")).putArray("layer_types").add("sliding_attention");
        assertThrows(IllegalArgumentException.class, () -> EmbeddingGemma2Config.fromJson(json));
    }

    @Test
    void usesDocumentedContextLimitRatherThanRawPositionLimit() throws Exception {
        EmbeddingGemma2Config config = EmbeddingGemma2Config.fromJson(validConfig());
        assertTrue(config.maxPositionEmbeddings() > config.maxInputTokens());
        assertEquals(8192, config.maxInputTokens());
    }

    private JsonNode validConfig() throws Exception {
        return mapper.readTree("""
                {
                  "model_type":"embedding_gemma2",
                  "text_config":{
                    "model_type":"embedding_gemma2_text",
                    "hidden_size":512,
                    "intermediate_size":2048,
                    "num_hidden_layers":24,
                    "num_attention_heads":4,
                    "num_key_value_heads":2,
                    "vocab_size":262144,
                    "max_position_embeddings":262144,
                    "dtype":"bfloat16",\n                    "head_dim":256,\n                    "embedding_dim":768,
                    "rms_norm_eps":0.000001,
                    "rope_parameters":{\n                      "full_attention":{"rope_theta":1000000.0,"rope_type":"default"},\n                      "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}\n                    },
                    "sliding_window":512,
                    "per_layer_config":{
                      "5":{"head_dim":256,"num_key_value_heads":1},
                      "23":{"head_dim":256,"num_key_value_heads":1}
                    }
                  }
                }
                """);
    }
}
