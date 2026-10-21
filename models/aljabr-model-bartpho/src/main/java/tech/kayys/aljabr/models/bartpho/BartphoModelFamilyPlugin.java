package tech.kayys.aljabr.models.bartpho;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BartphoModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bartpho",
                "BARTpho",
                List.of("bartpho"),
                List.of("BartForConditionalGeneration", "BartModel"),
                List.of(
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "sentencepiece_bpe_with_monolingual_vocab",
                        "direct_safetensor", "not_direct_runtime_tokenizer_only_over_bart_backbone",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bartpho-sentencepiece-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("sentencepiece.bpe.model", "dict.txt"),
                        List.of("tokenizer/sentencepiece.bpe.model", "tokenizer/dict.txt")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_bpe",
                        "status", "metadata_only_until_bartpho_tokenizer_runtime")));
    }
}
