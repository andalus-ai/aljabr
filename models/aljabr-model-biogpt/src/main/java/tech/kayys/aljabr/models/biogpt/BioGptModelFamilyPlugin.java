package tech.kayys.aljabr.models.biogpt;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BioGptModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "biogpt",
                "BioGPT",
                List.of("biogpt"),
                List.of("BioGptModel", "BioGptForCausalLM",
                        "BioGptForSequenceClassification", "BioGptForTokenClassification"),
                List.of(ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "moses_bpe_metadata_only",
                        "direct_safetensor", "pending_biogpt_decoder_and_moses_bpe_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "biogpt-moses-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.json", "merges.txt"),
                        List.of("tokenizer/vocab.json", "tokenizer/merges.txt")),
                Map.of(
                        "pre_tokenizer", "moses",
                        "status", "metadata_only_until_moses_bpe_runtime")));
    }
}
