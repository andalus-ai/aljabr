package tech.kayys.aljabr.models.embeddinggemma2.input;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

class EmbeddingGemma2TokenizerTest {
    private static final String CONFIG = """
            {
              "model_type":"embedding_gemma2",
              "text_config":{
                "model_type":"embedding_gemma2_text",
                "hidden_size":2,"intermediate_size":4,"num_hidden_layers":1,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,
                "embedding_dim":2,"vocab_size":16,"max_position_embeddings":8,
                "rms_norm_eps":0.000001,"dtype":"float32",
                "rope_parameters":{
                  "full_attention":{"rope_theta":10000},
                  "sliding_attention":{"rope_theta":10000}
                }
              }
            }
            """;

    private EmbeddingGemma2Config config() throws Exception {
        return EmbeddingGemma2Config.fromJson(new ObjectMapper().readTree(CONFIG));
    }

    @Test
    void addsBosAndBuildsUnpaddedMaskAndPositions() throws Exception {
        EmbeddingGemma2Tokenizer adapter = new EmbeddingGemma2Tokenizer(config(), new StubTokenizer());
        var result = adapter.tokenize("formatted input");

        assertArrayEquals(new int[] {1, 4, 5}, result.inputIds());
        assertArrayEquals(new boolean[] {true, true, true}, result.validTokens());
        assertArrayEquals(new int[] {0, 1, 2}, result.positionIds());
    }

    @Test
    void appliesTaskFormattingBeforeEncoding() throws Exception {
        StubTokenizer tokenizer = new StubTokenizer();
        EmbeddingGemma2Tokenizer adapter = new EmbeddingGemma2Tokenizer(config(), tokenizer);

        adapter.tokenize("cats", null, EmbeddingTask.RETRIEVAL_QUERY, EmbeddingGemma2InputRole.QUERY);

        assertEquals("task: search result | query: cats", tokenizer.lastEncodedText);
    }

    @Test
    void rejectsSequencesThatExceedModelLimit() throws Exception {
        EmbeddingGemma2Config shortConfig = new EmbeddingGemma2Config(
                config().modelType(), config().textModelType(), config().hiddenSize(),
                config().intermediateSize(), config().numHiddenLayers(), config().numAttentionHeads(),
                config().numKeyValueHeads(), config().headDim(), config().hiddenSizePerLayerInput(),
                config().hiddenActivation(), config().embeddingDim(), config().vocabSize(),
                2, 2, config().dtype(), config().rmsNormEps(), config().slidingWindow(),
                config().layerTypes(), config().perLayerConfig(), config().ropeParameters());
        EmbeddingGemma2Tokenizer adapter = new EmbeddingGemma2Tokenizer(shortConfig, new StubTokenizer());

        assertThrows(IllegalArgumentException.class, () -> adapter.tokenize("too many tokens"));
    }

    @Test
    void tokenizedArraysAreDefensivelyCopied() throws Exception {
        var result = new EmbeddingGemma2Tokenizer(config(), new StubTokenizer()).tokenize("x");
        int[] ids = result.inputIds();
        ids[0] = 9;
        assertEquals(1, result.inputIds()[0]);
    }

    private static final class StubTokenizer implements Tokenizer {
        private String lastEncodedText;
        @Override public long[] encode(String text, EncodeOptions options) {
            lastEncodedText = text;
            return text.equals("formatted input") ? new long[] {4, 5} : new long[] {4, 5};
        }
        @Override public String decode(long[] tokens, DecodeOptions options) { return "decoded"; }
        @Override public int vocabSize() { return 16; }
        @Override public int bosTokenId() { return 1; }
        @Override public int eosTokenId() { return 2; }
        @Override public int padTokenId() { return 0; }
        @Override public int[] allStopTokenIds() { return new int[] {2}; }
    }
}
