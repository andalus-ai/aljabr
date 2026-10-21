package tech.kayys.aljabr.models.mixtral;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MixtralModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "mixtral",
                "Mistral AI Mixtral",
                List.of("mixtral"),
                List.of("MixtralForCausalLM", "MixtralModel",
                        "MixtralForSequenceClassification", "MixtralForTokenClassification",
                        "MixtralForQuestionAnswering"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "mistral_sentencepiece_or_hf_bpe",
                        "direct_safetensor", "pending_mixtral_moe_routing_runtime_policy",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.sentencePieceBpe("mixtral-spm-bpe"),
                ModelTokenizerDescriptor.huggingFaceBpe("mixtral-hf-bpe"));
    }
}
