package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionException;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2AttentionBlock;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2EncoderLayer;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2InputRole;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2Tokenizer;
import tech.kayys.aljabr.spi.embedding.EmbeddingRequest;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

class EmbeddingGemma2TextModelRunnerTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void adaptsBatchRequestsAndPreservesRequestMetadata() throws Exception {
        var runner = new EmbeddingGemma2TextModelRunner(model());
        var request = new EmbeddingRequest("request-1", "embedding_gemma2",
                List.of("first query", "second query"), Map.of(),
                EmbeddingTask.RETRIEVAL_QUERY, 128);

        var response = runner.embed(request).toCompletableFuture().join();

        assertEquals("embedding_gemma2", runner.modelFamily());
        assertEquals("request-1", response.requestId());
        assertEquals("embedding_gemma2", response.model());
        assertEquals(2, response.embeddings().size());
        assertEquals(128, response.dimension());
        assertEquals(128, response.embeddings().get(0).length);
        assertEquals("RETRIEVAL_QUERY", response.metadata().get("task"));
    }

    @Test
    void rejectsInvalidTitleParameterAsFailedStage() throws Exception {
        var runner = new EmbeddingGemma2TextModelRunner(model());
        var request = new EmbeddingRequest("request-2", "embedding_gemma2",
                List.of("document"), Map.of("titles", List.of("one", "extra")),
                EmbeddingTask.RETRIEVAL_DOCUMENT, null);

        assertThrows(CompletionException.class,
                () -> runner.embed(request).toCompletableFuture().join());
    }

    @Test
    void mapsTaskIntentToTheCorrectInputRole() {
        assertEquals(EmbeddingGemma2InputRole.QUERY,
                EmbeddingGemma2TextModelRunner.roleFor(EmbeddingTask.RETRIEVAL_QUERY));
        assertEquals(EmbeddingGemma2InputRole.DOCUMENT,
                EmbeddingGemma2TextModelRunner.roleFor(EmbeddingTask.RETRIEVAL_DOCUMENT));
        assertEquals(EmbeddingGemma2InputRole.SINGLE,
                EmbeddingGemma2TextModelRunner.roleFor(EmbeddingTask.SEMANTIC_SIMILARITY));
    }

    private static EmbeddingGemma2TextModel model() throws Exception {
        var config = new EmbeddingGemma2Config(
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
        var attention = new EmbeddingGemma2AttentionBlock.Weights(
                matrix2(), null, matrix2(), null, matrix2(), null,
                matrix2(), null, vector2(), vector2());
        var mlp = new EmbeddingGemma2Mlp.Weights(matrix2(), matrix2(), matrix2());
        var encoder = new EmbeddingGemma2EncoderLayer.Weights(
                vector2(), vector2(), vector2(), vector2(), matrix2(), matrix2(), vector2());
        var layer = new EmbeddingGemma2TextBackbone.LayerWeights(attention, mlp, encoder);
        float[][] vocabulary = new float[16][2];
        for (float[] row : vocabulary) { row[0] = 1f; row[1] = -1f; }
        float[][] projection = new float[768][2];
        for (float[] row : projection) row[0] = 1f;
        var weights = new EmbeddingGemma2TextBackbone.Weights(
                vocabulary, matrix2(), vector2(), List.of(layer), vector2(), projection);
        return new EmbeddingGemma2TextModel(config,
                new EmbeddingGemma2Tokenizer(config, new FixtureTokenizer()),
                new EmbeddingGemma2TextBackbone(config, weights));
    }

    private static float[][] matrix2() { return new float[][]{{0f, 0f}, {0f, 0f}}; }
    private static float[] vector2() { return new float[]{1f, 1f}; }

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
