package tech.kayys.aljabr.models.functiongemma;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * FunctionGemma family registration.
 *
 * <p>The upstream checkpoint declares model_type=gemma3_text and
 * Gemma3ForCausalLM, so config-only resolution cannot distinguish it from
 * ordinary Gemma 3 text checkpoints. This plugin recognizes the dedicated
 * FunctionGemma control-token inventory in tokenizer_config.json.</p>
 */
@ApplicationScoped
public class FunctionGemmaModelFamilyPlugin implements ModelFamilyPlugin {

    private static final List<String> REQUIRED_PROTOCOL_TOKENS = List.of(
            FunctionGemmaTokenizer.START_FUNCTION_DECLARATION,
            FunctionGemmaTokenizer.END_FUNCTION_DECLARATION,
            FunctionGemmaTokenizer.START_FUNCTION_CALL,
            FunctionGemmaTokenizer.END_FUNCTION_CALL,
            FunctionGemmaTokenizer.START_FUNCTION_RESPONSE,
            FunctionGemmaTokenizer.END_FUNCTION_RESPONSE,
            FunctionGemmaTokenizer.ESCAPE);

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "functiongemma",
                "Google FunctionGemma",
                List.of("functiongemma"),
                List.of("Gemma3ForCausalLM"),
                List.of(ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "optional",
                        "base_architecture", "gemma3_text",
                        "upstream_config_model_type", "gemma3_text",
                        "tokenizer", "functiongemma-gemma-bpe",
                        "chat_protocol", "functiongemma",
                        "function_calling", "native_control_tokens",
                        "default_checkpoint", "google/functiongemma-270m-it",
                        "inference_scope", "uses_shared_gemma3_text_execution",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public boolean matchesCheckpoint(Path modelDirectory) {
        if (modelDirectory == null) {
            return false;
        }
        Path configPath = modelDirectory.resolve("config.json");
        Path tokenizerConfigPath = modelDirectory.resolve("tokenizer_config.json");
        Path tokenizerJsonPath = modelDirectory.resolve("tokenizer.json");
        if (!Files.isRegularFile(configPath)
                || (!Files.isRegularFile(tokenizerConfigPath) && !Files.isRegularFile(tokenizerJsonPath))) {
            return false;
        }
        try {
            ModelConfig config = ModelConfig.load(configPath, new ObjectMapper());
            if (!"gemma3_text".equals(config.modelType())
                    || !"Gemma3ForCausalLM".equals(config.primaryArchitecture())) {
                return false;
            }
            String tokenizerConfig = Files.isRegularFile(tokenizerConfigPath)
                    ? Files.readString(tokenizerConfigPath) : "";
            String tokenizerJson = Files.isRegularFile(tokenizerJsonPath)
                    ? Files.readString(tokenizerJsonPath) : "";
            return REQUIRED_PROTOCOL_TOKENS.stream()
                    .allMatch(token -> tokenizerConfig.contains(token) || tokenizerJson.contains(token));
        } catch (IOException | RuntimeException ignored) {
            return false;
        }
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new FunctionGemmaArchitecture());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("functiongemma-gemma-bpe"));
    }
}
