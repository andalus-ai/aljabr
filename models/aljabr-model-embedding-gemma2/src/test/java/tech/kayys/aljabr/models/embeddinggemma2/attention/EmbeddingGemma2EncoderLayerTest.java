package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfig;

class EmbeddingGemma2EncoderLayerTest {

    @Test
    void executesAllThreeResidualSubBlocksAndPreservesInput() throws Exception {
        var json = new ObjectMapper().readTree("""
                {
                  "model_type":"embedding_gemma2",
                  "text_config":{
                    "model_type":"embedding_gemma2_text","hidden_size":2,"intermediate_size":2,
                    "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
                    "head_dim":2,"hidden_size_per_layer_input":2,"embedding_dim":2,"vocab_size":16,"max_position_embeddings":16,
                    "rms_norm_eps":0.000001,"dtype":"float32",
                    "layer_types":["full_attention"],
                    "rope_parameters":{
                      "full_attention":{"rope_theta":10000,"rope_type":"default"},
                      "sliding_attention":{"rope_theta":10000,"rope_type":"default"}
                    }
                  }
                }
                """);
        var config = EmbeddingGemma2Config.fromJson(json);
        var layer = new EmbeddingGemma2LayerConfig(0,
                EmbeddingGemma2LayerConfig.AttentionType.FULL, 1, 1, 2,
                10000d, "default", null, null);
        float[][] identity = {{1f, 0f}, {0f, 1f}};
        var attention = new EmbeddingGemma2AttentionBlock.Weights(
                identity, null, identity, null, identity, null, new float[][]{{0f, 0f}, {0f, 0f}}, null,
                new float[]{1f, 1f}, new float[]{1f, 1f});
        var mlp = new EmbeddingGemma2Mlp.Weights(
                new float[][]{{0f, 0f}, {0f, 0f}},
                new float[][]{{0f, 0f}, {0f, 0f}},
                new float[][]{{0f, 0f}, {0f, 0f}});
        var weights = new EmbeddingGemma2EncoderLayer.Weights(
                new float[]{1f, 1f}, new float[]{1f, 1f},
                new float[]{1f, 1f}, new float[]{1f, 1f},
                new float[][]{{0f, 0f}, {0f, 0f}},
                new float[][]{{0f, 0f}, {0f, 0f}},
                new float[]{1f, 1f});
        var encoder = new EmbeddingGemma2EncoderLayer(config, layer, attention, mlp, weights);

        float[][] input = {{1f, 2f}, {3f, 4f}};
        float[][] original = {{1f, 2f}, {3f, 4f}};
        float[][] result = encoder.compute(input, new float[][]{{0f, 0f}, {0f, 0f}},
                new int[]{0, 1}, null);

        assertArrayEquals(original[0], input[0]);
        assertArrayEquals(original[1], input[1]);
        assertArrayEquals(original[0], result[0], 1e-5f);
        assertArrayEquals(original[1], result[1], 1e-5f);
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertTrue(Float.isFinite(result[0][0]));
        assertTrue(Float.isFinite(result[1][1]));
    }
}
