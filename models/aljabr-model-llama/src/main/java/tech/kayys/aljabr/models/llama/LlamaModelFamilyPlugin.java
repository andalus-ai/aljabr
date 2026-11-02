package tech.kayys.aljabr.models.llama;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.models.llama.LLaMA3Family;
import tech.kayys.aljabr.models.llama.LlamaFamily;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LlamaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "llama",
                "Meta Llama",
                List.of("llama", "llama3"),
                List.of("LlamaForCausalLM"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                Map.of(
                        "bundle_profile", "core",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new LlamaFamily(), new LLaMA3Family());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("llama-spm-bpe"));
    }
}
