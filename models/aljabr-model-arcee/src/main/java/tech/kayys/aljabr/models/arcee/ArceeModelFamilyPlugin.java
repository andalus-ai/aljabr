package tech.kayys.aljabr.models.arcee;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ArceeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "arcee",
                "Arcee",
                List.of("arcee"),
                List.of("ArceeForCausalLM", "ArceeModel", "ArceeForSequenceClassification",
                        "ArceeForQuestionAnswering", "ArceeForTokenClassification"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "llama_compatible_tokenizer",
                        "direct_safetensor", "pending_arcee_llama_variant_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.sentencePieceBpe("arcee-spm-bpe"),
                ModelTokenizerDescriptor.huggingFaceBpe("arcee-hf-bpe"));
    }
}
