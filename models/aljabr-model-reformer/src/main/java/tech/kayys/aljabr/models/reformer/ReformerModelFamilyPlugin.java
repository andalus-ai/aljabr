package tech.kayys.aljabr.models.reformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ReformerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "reformer",
                "Reformer",
                List.of("reformer"),
                List.of("ReformerModel", "ReformerModelWithLMHead", "ReformerForMaskedLM",
                        "ReformerForSequenceClassification", "ReformerForQuestionAnswering"),
                List.of(ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "sentencepiece_bpe_metadata_only",
                        "direct_safetensor", "pending_reformer_lsh_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "reformer-sentencepiece-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("spiece.model"),
                        List.of("tokenizer/tokenizer.json"), List.of("tokenizer/spiece.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_metaspace",
                        "status", "metadata_only_until_reformer_bpe_runtime")));
    }
}
