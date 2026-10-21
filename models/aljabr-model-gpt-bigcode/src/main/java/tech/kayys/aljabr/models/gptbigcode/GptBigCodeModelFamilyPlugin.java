package tech.kayys.aljabr.models.gptbigcode;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class GptBigCodeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "gpt_bigcode",
                "GPT-BigCode / StarCoder",
                List.of("gpt_bigcode", "starcoder2"),
                List.of("GPTBigCodeForCausalLM", "GPTBigCodeModel", "GPTBigCodeForSequenceClassification",
                        "GPTBigCodeForTokenClassification", "Starcoder2ForCausalLM", "Starcoder2Model",
                        "Starcoder2ForSequenceClassification", "Starcoder2ForTokenClassification"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "byte_level_bpe",
                        "direct_safetensor", "pending_bigcode_multi_query_and_starcoder2_layout_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("bigcode-byte-level-bpe"));
    }
}
