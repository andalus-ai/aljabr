package tech.kayys.aljabr.safetensor.functiongemma;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;

import java.util.LinkedHashSet;
import java.util.Set;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FunctionGemmaSafetensorCheckpointLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void acceptsTheCompleteGemma3TextTensorLayout() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        ModelConfig config = mapper.readValue("""
                {
                  "model_type": "gemma3_text",
                  "architectures": ["Gemma3ForCausalLM"],
                  "num_hidden_layers": 2
                }
                """, ModelConfig.class);
        FunctionGemmaArchitecture architecture = new FunctionGemmaArchitecture();
        Set<String> tensors = new LinkedHashSet<>();
        tensors.add(architecture.embedTokensWeight());
        tensors.add(architecture.finalNormWeight());
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            tensors.add(architecture.layerQueryWeight(layer));
            tensors.add(architecture.layerKeyWeight(layer));
            tensors.add(architecture.layerValueWeight(layer));
            tensors.add(architecture.layerOutputWeight(layer));
            tensors.add(architecture.layerAttentionNormWeight(layer));
            tensors.add(architecture.layerPostAttnNormWeight(layer));
            tensors.add(architecture.layerPreFfnNormWeight(layer));
            tensors.add(architecture.layerPostFfnNormWeight(layer));
            tensors.add(architecture.layerQueryNormWeight(layer));
            tensors.add(architecture.layerKeyNormWeight(layer));
            tensors.add(architecture.layerFfnGateWeight(layer));
            tensors.add(architecture.layerFfnUpWeight(layer));
            tensors.add(architecture.layerFfnDownWeight(layer));
        }

        FunctionGemmaSafetensorCheckpointLoader.validateTensorLayout(
                tensors, config, architecture);
    }

    @Test
    void reportsMissingRequiredWeightsBeforeReturningCheckpoint() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "gemma3_text",
                  "architectures": ["Gemma3ForCausalLM"],
                  "num_hidden_layers": 1
                }
                """, ModelConfig.class);

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> FunctionGemmaSafetensorCheckpointLoader.validateTensorLayout(
                        Set.of(), config, new FunctionGemmaArchitecture()));
        assertTrue(failure.getMessage().contains("missing 15 required tensors"));
    }

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
