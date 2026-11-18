package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;

class EmbeddingGemma2TextBackboneTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void composesPleEncoderFinalNormAndTokenProjection() throws Exception {
        EmbeddingGemma2Config config = config();
        EmbeddingGemma2TextBackbone backbone = new EmbeddingGemma2TextBackbone(config, weights());

        float[][] input = {{1.0f, -1.0f}};
        float[][] output = backbone.encode(input, new int[]{0}, null);

        assertEquals(1, output.length);
        assertEquals(3, output[0].length);
        assertEquals(1, backbone.layerCount());
        assertEquals(2, backbone.hiddenSize());
        assertEquals(3, backbone.embeddingDimension());
        assertNotSame(input[0], output[0]);
        assertEquals(1.0f, input[0][0]);
        assertEquals(-1.0f, input[0][1]);
    }

    @Test
    void rejectsMissingLayerWeights() throws Exception {
        EmbeddingGemma2TextBackbone.Weights valid = weights();
        EmbeddingGemma2TextBackbone.Weights missing = new EmbeddingGemma2TextBackbone.Weights(
                valid.pleProjection(), valid.pleNormScale(), List.of(),
                valid.finalNormScale(), valid.embeddingProjection());
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2TextBackbone(config(), missing));
    }

    @Test
    void rejectsPositionLengthMismatch() throws Exception {
        EmbeddingGemma2TextBackbone backbone =
                new EmbeddingGemma2TextBackbone(config(), weights());
        assertThrows(IllegalArgumentException.class,
                () -> backbone.encode(new float[][]{{1f, 2f}}, new int[0], null));
    }

    private static EmbeddingGemma2Config config() throws Exception {
        return new EmbeddingGemma2Config(
                EmbeddingGemma2Config.MODEL_TYPE,
                EmbeddingGemma2Config.TEXT_MODEL_TYPE,
                2, 2, 1, 1, 1, 2, 2,
                EmbeddingGemma2Config.DEFAULT_HIDDEN_ACTIVATION,
                3, 16, 16, 8192, "float32", 1.0e-6,
                null, List.of("full_attention"), Map.of(),
                JSON.readTree("""
                    {
                      "full_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
                    }
                    """));
    }

    private static EmbeddingGemma2TextBackbone.Weights weights() {
        float[][] matrix2 = {{0f, 0f}, {0f, 0f}};
        float[] vector2 = {1f, 1f};
        EmbeddingGemma2AttentionBlock.Weights attention =
                new EmbeddingGemma2AttentionBlock.Weights(
                        matrix2, null, matrix2, null, matrix2, null,
                        matrix2, null, vector2, vector2);
        EmbeddingGemma2Mlp.Weights mlp =
                new EmbeddingGemma2Mlp.Weights(matrix2, matrix2, matrix2);
        EmbeddingGemma2EncoderLayer.Weights encoder =
                new EmbeddingGemma2EncoderLayer.Weights(
                        vector2, vector2, vector2, vector2,
                        matrix2, matrix2, vector2);
        EmbeddingGemma2TextBackbone.LayerWeights layer =
                new EmbeddingGemma2TextBackbone.LayerWeights(attention, mlp, encoder);
        return new EmbeddingGemma2TextBackbone.Weights(
                matrix2, vector2, List.of(layer), vector2,
                new float[][]{{0f, 0f}, {0f, 0f}, {0f, 0f}});
    }
}
