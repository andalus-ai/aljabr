package tech.kayys.aljabr.models.functiongemma;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

/**
 * FunctionGemma family registration.
 *
 * <p>The model reuses the Gemma 3 text decoder architecture. Its distinct
 * identity is expressed through this family adapter, tokenizer routing and
 * chat template rather than a duplicated transformer implementation.</p>
 */
@ApplicationScoped
public class FunctionGemmaModelFamilyPlugin implements ModelFamilyPlugin {

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
                        "tokenizer", "functiongemma-gemma-bpe",
                        "chat_protocol", "functiongemma",
                        "function_calling", "native_control_tokens",
                        "default_checkpoint", "google/functiongemma-270m-it",
                        "inference_scope", "uses_shared_gemma3_text_execution",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new FunctionGemmaArchitecture());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("functiongemma-gemma-bpe"));
    }
}
