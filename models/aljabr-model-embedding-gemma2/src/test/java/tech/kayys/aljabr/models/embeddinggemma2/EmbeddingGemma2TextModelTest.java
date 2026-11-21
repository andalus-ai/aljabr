package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2AttentionBlock;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2EncoderLayer;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2InputRole;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2Tokenizer;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

class EmbeddingGemma2TextModelTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void composesSharedTokenizerTextBackboneAndMatryoshkaOutput() throws Exception {
        var config = config();
        var tokenizer = new EmbeddingGemma2Tokenizer(config, new FixtureTokenizer());
        var model = new EmbeddingGemma2TextModel(config, tokenizer,
                new EmbeddingGemma2TextBackbone(config, weights()));

        float[] vector = model.embed(
                "find related documents", null,
                EmbeddingTask.RETRIEVAL_QUERY, EmbeddingGemma2InputRole.QUERY, 128);

        assertEquals(128, vector.length);
        double norm = 0.0;
        for (float value : vector) norm += (double) value * value;
        assertTrue(Math.abs(Math.sqrt(norm) - 1.0) < 1.0e-5);
    }

    @Test
    void rejectsUnsupportedMatryoshkaDimension() throws Exception {
        var config = config();
        var model = new EmbeddingGemma2TextModel(config,
                new EmbeddingGemma2Tokenizer(config, new FixtureTokenizer()),
                new EmbeddingGemma2TextBackbone(config, weights()));

        assertThrows(IllegalArgumentException.class, () -> model.embed(
                "query", null, EmbeddingTask.RETRIEVAL_QUERY,
                EmbeddingGemma2InputRole.QUERY, 384));
    }

    private static EmbeddingGemma2Config config() throws Exception {
        return new EmbeddingGemma2Config(
                EmbeddingGemma2Config.MODEL_TYPE,
                EmbeddingGemma2Config.TEXT_MODEL_TYPE,
                2, 2, 1, 1, 1, 2, 2,
                EmbeddingGemma2Config.DEFAULT_HIDDEN_ACTIVATION,
                768, 16, 16, 8192, "float32", 1.0e-6,
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
        var attention = new EmbeddingGemma2AttentionBlock.Weights(
                matrix2, null, matrix2, null, matrix2, null,
                matrix2, null, vector2, vector2);
        var mlp = new EmbeddingGemma2Mlp.Weights(matrix2, matrix2, matrix2);
        var encoder = new EmbeddingGemma2EncoderLayer.Weights(
                vector2, vector2, vector2, vector2, matrix2, matrix2, vector2);
        var layer = new EmbeddingGemma2TextBackbone.LayerWeights(attention, mlp, encoder);
        float[][] vocabulary = new float[16][2];
        for (float[] row : vocabulary) {
            row[0] = 1.0f;
            row[1] = -1.0f;
        }
        float[][] projection = new float[768][2];
        for (float[] row : projection) row[0] = 1.0f;
        return new EmbeddingGemma2TextBackbone.Weights(
                vocabulary, matrix2, vector2, List.of(layer), vector2, projection);
    }

    private static final class FixtureTokenizer implements Tokenizer {
        @Override public long[] encode(String text, EncodeOptions options) { return new long[]{1, 2}; }
        @Override public String decode(long[] tokens, DecodeOptions options) { return ""; }
        @Override public int vocabSize() { return 16; }
        @Override public int bosTokenId() { return 0; }
        @Override public int eosTokenId() { return 1; }
        @Override public int padTokenId() { return 0; }
        @Override public int[] allStopTokenIds() { return new int[]{1}; }
    }
}
