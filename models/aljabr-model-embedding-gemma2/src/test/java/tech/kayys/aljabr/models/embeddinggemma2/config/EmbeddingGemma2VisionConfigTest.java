package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class EmbeddingGemma2VisionConfigTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void parsesPublishedEmbeddingGemma2NestedImageProcessorMetadata() throws Exception {
        EmbeddingGemma2VisionConfig config = EmbeddingGemma2VisionConfig.fromJson(
                modelConfig(), processorConfig());

        assertEquals("gemma4_vision", config.modelType());
        assertEquals(768, config.hiddenSize());
        assertEquals(3072, config.intermediateSize());
        assertEquals(16, config.numHiddenLayers());
        assertEquals(12, config.numAttentionHeads());
        assertEquals(12, config.numKeyValueHeads());
        assertEquals(64, config.headDim());
        assertEquals(16, config.patchSize());
        assertEquals(10240, config.positionEmbeddingSize());
        assertEquals(3, config.poolingKernelSize());
        assertEquals(280, config.maxSoftTokens());
        assertEquals(100.0, config.ropeTheta());
    }

    @Test
    void readsSoftTokenBudgetFromNestedImageProcessor() throws Exception {
        JsonNode processor = processorConfig();
        ((com.fasterxml.jackson.databind.node.ObjectNode) processor.path("image_processor"))
                .put("max_soft_tokens", 140);

        EmbeddingGemma2VisionConfig config = EmbeddingGemma2VisionConfig.fromJson(modelConfig(), processor);
        assertEquals(140, config.maxSoftTokens());
    }

    @Test
    void rejectsPatchGeometryMismatchInNestedImageProcessor() throws Exception {
        JsonNode processor = processorConfig();
        ((com.fasterxml.jackson.databind.node.ObjectNode) processor.path("image_processor"))
                .put("patch_size", 14);

        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionConfig.fromJson(modelConfig(), processor));
    }

    @Test
    void rejectsPoolingGeometryMismatchInNestedImageProcessor() throws Exception {
        JsonNode processor = processorConfig();
        ((com.fasterxml.jackson.databind.node.ObjectNode) processor.path("image_processor"))
                .put("pooling_kernel_size", 2);

        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionConfig.fromJson(modelConfig(), processor));
    }

    private JsonNode modelConfig() throws Exception {
        return mapper.readTree("""
                {
                  "model_type": "embedding_gemma2",
                  "vision_config": {
                    "model_type": "gemma4_vision",
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 16,
                    "num_attention_heads": 12,
                    "num_key_value_heads": 12,
                    "head_dim": 64,
                    "patch_size": 16,
                    "position_embedding_size": 10240,
                    "pooling_kernel_size": 3,
                    "rms_norm_eps": 0.000001,
                    "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"},
                    "standardize": false,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "use_clipped_linears": false
                  }
                }
                """);
    }

    private JsonNode processorConfig() throws Exception {
        return mapper.readTree("""
                {
                  "processor_class": "EmbeddingGemma2Processor",
                  "image_processor": {
                    "image_processor_type": "Gemma4ImageProcessor",
                    "do_convert_rgb": true,
                    "do_normalize": false,
                    "do_rescale": true,
                    "do_resize": true,
                    "max_soft_tokens": 280,
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "rescale_factor": 0.00392156862745098,
                    "resample": 3
                  }
                }
                """);
    }
}
