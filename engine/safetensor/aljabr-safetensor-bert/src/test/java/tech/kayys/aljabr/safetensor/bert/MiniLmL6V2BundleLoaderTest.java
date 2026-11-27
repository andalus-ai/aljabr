package tech.kayys.aljabr.safetensor.bert;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;

class MiniLmL6V2BundleLoaderTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void rejectsNonMeanPoolingProfileBeforeLoadingTokenizerOrWeights() throws Exception {
        writeMetadata(
                """
                {
                  "word_embedding_dimension": 384,
                  "pooling_mode_mean_tokens": false,
                  "pooling_mode_cls_token": false,
                  "pooling_mode_max_tokens": false,
                  "pooling_mode_mean_sqrt_len_tokens": false,
                  "pooling_mode_lasttoken": false
                }
                """,
                validModules());

        IOException failure = assertThrows(IOException.class, () -> loader().load(
                temporaryDirectory, new tech.kayys.aljabr.backend.cpu.CpuBackend()));

        assertTrue(failure.getMessage().contains("Unsupported MiniLM pooling configuration"));
    }

    @Test
    void rejectsUnsupportedSentenceTransformerModuleSequence() throws Exception {
        writeMetadata(validPooling(), """
                [
                  {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
                  {"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
                  {"idx":2,"name":"2","path":"2_Dense","type":"sentence_transformers.models.Dense"}
                ]
                """);

        IOException failure = assertThrows(IOException.class, () -> loader().load(
                temporaryDirectory, new tech.kayys.aljabr.backend.cpu.CpuBackend()));

        assertTrue(failure.getMessage().contains("expected Transformer, Pooling, Normalize"));
    }

    private MiniLmL6V2BundleLoader loader() {
        return new MiniLmL6V2BundleLoader(
                new ObjectMapper(), new BertSafetensorLoader(new SafetensorShardLoader()));
    }

    private void writeMetadata(String pooling, String modules) throws IOException {
        Files.createDirectories(temporaryDirectory.resolve("1_Pooling"));
        Files.writeString(temporaryDirectory.resolve("config.json"), """
                {
                  "model_type": "bert",
                  "vocab_size": 30522,
                  "hidden_size": 384,
                  "num_hidden_layers": 6,
                  "num_attention_heads": 12,
                  "intermediate_size": 1536,
                  "max_position_embeddings": 512,
                  "type_vocab_size": 2,
                  "pad_token_id": 0,
                  "layer_norm_eps": 1.0e-12,
                  "hidden_act": "gelu"
                }
                """);
        Files.writeString(temporaryDirectory.resolve("sentence_bert_config.json"),
                "{\"max_seq_length\":256}");
        Files.writeString(temporaryDirectory.resolve("1_Pooling/config.json"), pooling);
        Files.writeString(temporaryDirectory.resolve("modules.json"), modules);
    }

    private static String validPooling() {
        return """
                {
                  "word_embedding_dimension": 384,
                  "pooling_mode_mean_tokens": true,
                  "pooling_mode_cls_token": false,
                  "pooling_mode_max_tokens": false,
                  "pooling_mode_mean_sqrt_len_tokens": false,
                  "pooling_mode_lasttoken": false
                }
                """;
    }

    private static String validModules() {
        return """
                [
                  {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
                  {"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
                  {"idx":2,"name":"2","path":"2_Normalize","type":"sentence_transformers.models.Normalize"}
                ]
                """;
    }
}
