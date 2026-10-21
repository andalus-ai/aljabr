package tech.kayys.aljabr.models.zamba;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ZambaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "zamba",
                "Zamba / Zamba2",
                List.of("zamba", "zamba2"),
                List.of("ZambaForCausalLM", "ZambaModel", "ZambaForSequenceClassification",
                        "Zamba2ForCausalLM", "Zamba2Model", "Zamba2ForSequenceClassification"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "hybrid_llama_compatible_tokenizer",
                        "direct_safetensor", "pending_zamba_hybrid_attention_mamba_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.sentencePieceBpe("zamba-spm-bpe"),
                ModelTokenizerDescriptor.huggingFaceBpe("zamba-hf-bpe"));
    }
}
