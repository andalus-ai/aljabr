package tech.kayys.aljabr.models.deberta;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DebertaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "deberta",
                "DeBERTa",
                List.of("deberta"),
                List.of("DebertaModel", "DebertaForMaskedLM", "DebertaForSequenceClassification"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "byte_level_bpe",
                        "direct_safetensor", "not_causal_lm",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                new ModelTokenizerDescriptor(
                        "deberta-byte-level-bpe",
                        ModelTokenizerKind.HUGGING_FACE_BPE,
                        List.of(
                                List.of("vocab.json", "merges.txt"),
                                List.of("tokenizer/vocab.json", "tokenizer/merges.txt")),
                        Map.of("pre_tokenizer", "gpt2")));
    }
}
