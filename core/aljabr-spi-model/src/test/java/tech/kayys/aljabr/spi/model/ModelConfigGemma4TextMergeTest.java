package tech.kayys.aljabr.spi.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelConfigGemma4TextMergeTest {

    @Test
    void mergesSlidingWindowMaxPositionsAndActivationFromTextConfig() throws IOException {
        String json = """
                {
                  "model_type": "gemma4",
                  "architectures": ["Gemma4ForConditionalGeneration"],
                  "dtype": "bfloat16",
                  "eos_token_id": [1, 106],
                  "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 2560,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "intermediate_size": 10240,
                    "vocab_size": 262144,
                    "sliding_window": 512,
                    "max_position_embeddings": 131072,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "head_dim": 256,
                    "layer_types": ["sliding_attention", "full_attention"]
                  }
                }
                """;
        Path dir = Files.createTempDirectory("aljabr-modelconfig-test");
        Path cfgPath = dir.resolve("config.json");
        Files.writeString(cfgPath, json, StandardCharsets.UTF_8);
        try {
            ModelConfig cfg = ModelConfig.load(cfgPath, new ObjectMapper());
            assertEquals(512, cfg.slidingWindowSize());
            assertEquals(131072, cfg.maxPositionEmbeddings());
            assertEquals("gelu_pytorch_tanh", cfg.hiddenAct());
            assertEquals(2560, cfg.hiddenSize());
            assertEquals(2, cfg.numHiddenLayers());
            assertTrue(cfg.isSlidingAttentionLayer(0));
            assertEquals("sliding_attention", cfg.layerType(0));
            assertEquals("full_attention", cfg.layerType(1));
        } finally {
            Files.deleteIfExists(cfgPath);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    void mapsCommunityGgufFamiliesToArchitectureClasses() {
        assertEquals("Gemma4ForConditionalGeneration", ggufPrimaryArchitecture("gemma4"));
        assertEquals("Gemma4ForCausalLM", ggufPrimaryArchitecture("gemma4_text"));
        assertEquals("Gemma4VisionModel", ggufPrimaryArchitecture("gemma4_vision"));
        assertEquals("Gemma4AudioModel", ggufPrimaryArchitecture("gemma4_audio"));
        assertEquals("Gemma4ForMultimodalLM", ggufPrimaryArchitecture("gemma4_unified"));
        assertEquals("YiForCausalLM", ggufPrimaryArchitecture("yi"));
        assertEquals("Cohere2ForCausalLM", ggufPrimaryArchitecture("cohere2"));
        assertEquals("DeepseekV3ForCausalLM", ggufPrimaryArchitecture("deepseek_v3"));
        assertEquals("KimiForCausalLM", ggufPrimaryArchitecture("kimi"));
    }

    @Test
    void mergesGemma4UnifiedTextConfigWithoutLosingWrapperIdentity() throws IOException {
        String json = """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4ForMultimodalLM"],
                  "processor_class": "Gemma4Processor",
                  "text_config": {
                    "model_type": "gemma4_unified_text",
                    "architectures": ["Gemma4ForCausalLM"],
                    "hidden_size": 3840,
                    "num_hidden_layers": 48,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 1,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "attention_k_eq_v": true,
                    "intermediate_size": 15360,
                    "vocab_size": 262144,
                    "vocab_size_per_layer_input": 262144,
                    "sliding_window": 1024,
                    "max_position_embeddings": 131072,
                    "layer_types": [%s]
                  },
                  "vision_config": {"model_type": "gemma4_unified_vision"},
                  "audio_config": {"model_type": "gemma4_unified_audio"}
                }
                """.formatted(gemma4TwelveBLayerTypesJson());
        Path dir = Files.createTempDirectory("aljabr-modelconfig-unified-test");
        Path cfgPath = dir.resolve("config.json");
        Files.writeString(cfgPath, json, StandardCharsets.UTF_8);
        try {
            ModelConfig cfg = ModelConfig.load(cfgPath, new ObjectMapper());
            assertEquals("gemma4_unified", cfg.modelType());
            assertEquals("Gemma4ForMultimodalLM", cfg.primaryArchitecture());
            assertEquals(3840, cfg.hiddenSize());
            assertEquals(48, cfg.numHiddenLayers());
            assertEquals(16, cfg.numAttentionHeads());
            assertEquals(8, cfg.numKeyValueHeads());
            assertEquals(8, cfg.resolvedNumKvHeadsForLayer(0));
            assertEquals(1, cfg.resolvedNumKvHeadsForLayer(5));
            assertEquals(256, cfg.resolvedHeadDimForLayer(0));
            assertEquals(512, cfg.resolvedHeadDimForLayer(5));
            assertEquals(1024, cfg.slidingWindowSize());
            assertEquals(131072, cfg.maxPositionEmbeddings());
            assertTrue(ModelRuntimeTraits.fallbackFromConfig(cfg).nativeBf16Matvec());
        } finally {
            Files.deleteIfExists(cfgPath);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    void mergesGemma4MoeAliasesFromTextConfig() throws IOException {
        String json = """
                {
                  "model_type": "gemma4",
                  "architectures": ["Gemma4ForConditionalGeneration"],
                  "text_config": {
                    "model_type": "gemma4_text",
                    "enable_moe_block": true,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 704,
                    "use_double_wide_mlp": true
                  }
                }
                """;
        Path dir = Files.createTempDirectory("aljabr-modelconfig-test");
        Path cfgPath = dir.resolve("config.json");
        Files.writeString(cfgPath, json, StandardCharsets.UTF_8);
        try {
            ModelConfig cfg = ModelConfig.load(cfgPath, new ObjectMapper());
            assertTrue(cfg.enableMoeBlock());
            assertEquals(128, cfg.numLocalExperts());
            assertEquals(8, cfg.numExpertsPerTok());
            assertEquals(704, cfg.moeIntermediateSize());
            assertTrue(cfg.usesDoubleWideMlp());
            assertTrue(cfg.isMoe());
            assertTrue(cfg.isGemma4PackedMoe());
            assertTrue(cfg.requiresGemma4PackedMoeRuntime());
        } finally {
            Files.deleteIfExists(cfgPath);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    void parsesGemma4RootMoeAliasesWithoutTextConfigMerge() throws IOException {
        String json = """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "enable_moe_block": true,
                  "num_experts": 128,
                  "top_k_experts": 8,
                  "moe_intermediate_size": 704
                }
                """;
        Path dir = Files.createTempDirectory("aljabr-modelconfig-root-moe-test");
        Path cfgPath = dir.resolve("config.json");
        Files.writeString(cfgPath, json, StandardCharsets.UTF_8);
        try {
            ModelConfig cfg = ModelConfig.load(cfgPath, new ObjectMapper());
            assertTrue(cfg.enableMoeBlock());
            assertEquals(128, cfg.numLocalExperts());
            assertEquals(8, cfg.numExpertsPerTok());
            assertEquals(704, cfg.moeIntermediateSize());
            assertTrue(cfg.isMoe());
            assertTrue(cfg.requiresGemma4PackedMoeRuntime());
        } finally {
            Files.deleteIfExists(cfgPath);
            Files.deleteIfExists(dir);
        }
    }

    private static String ggufPrimaryArchitecture(String architecture) {
        return ModelConfig.fromGgufMetadata(Map.of("general.architecture", architecture))
                .primaryArchitecture();
    }

    private static String gemma4TwelveBLayerTypesJson() {
        StringBuilder json = new StringBuilder();
        for (int layer = 0; layer < 48; layer++) {
            if (layer > 0) {
                json.append(", ");
            }
            json.append('"')
                    .append(layer > 0 && (layer + 1) % 6 == 0
                            ? "full_attention"
                            : "sliding_attention")
                    .append('"');
        }
        return json.toString();
    }
}
