package tech.kayys.aljabr.models.ministral3;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Ministral3ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "ministral3",
                "Mistral AI Ministral 3",
                List.of("ministral3", "ministral_3", "ministral-3"),
                List.of("Ministral3ForCausalLM", "Ministral3Model",
                        "Ministral3ForSequenceClassification", "Ministral3ForTokenClassification",
                        "Ministral3ForQuestionAnswering"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "mistral_tekken_or_hf_tokenizer_json",
                        "direct_safetensor", "pending_ministral3_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "ministral3-tekken-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json"), List.of("tekken.json")),
                Map.of(
                        "tokenizer", "mistral_tekken",
                        "status", "metadata_only")));
    }
}
