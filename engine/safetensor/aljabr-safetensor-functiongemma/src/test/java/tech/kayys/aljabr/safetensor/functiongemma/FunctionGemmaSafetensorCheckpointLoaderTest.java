package tech.kayys.aljabr.safetensor.functiongemma;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FunctionGemmaSafetensorCheckpointLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void rejectsOrdinaryGemma3BeforeOpeningSafetensors() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "gemma3_text",
                  "architectures": ["Gemma3ForCausalLM"],
                  "hidden_size": 640,
                  "num_hidden_layers": 18,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 1,
                  "intermediate_size": 2048,
                  "vocab_size": 262144
                }
                """);
        Files.writeString(tempDir.resolve("tokenizer_config.json"), """
                {"added_tokens_decoder": {}}
                """);

        SafetensorShardLoader mustNotOpen = new SafetensorShardLoader() {
            @Override
            public SafetensorShardSession open(Path modelPath) {
                throw new AssertionError("weights must not be opened for a non-FunctionGemma checkpoint");
            }
        };
        FunctionGemmaSafetensorCheckpointLoader loader =
                new FunctionGemmaSafetensorCheckpointLoader(mustNotOpen, new ObjectMapper());

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> loader.load(tempDir));
        assertTrue(failure.getMessage().contains("not a recognized FunctionGemma checkpoint"));
    }
}
