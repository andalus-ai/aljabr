package tech.kayys.aljabr.models.phobert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class PhoBertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "phobert",
                "PhoBERT",
                List.of("phobert"),
                List.of("RobertaModel", "RobertaForMaskedLM",
                        "RobertaForSequenceClassification", "RobertaForTokenClassification",
                        "RobertaForQuestionAnswering"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "phobert_bpe_vocab_and_codes",
                        "direct_safetensor", "roberta_backbone_tokenizer_only_metadata",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "phobert-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.txt", "bpe.codes"),
                        List.of("tokenizer/vocab.txt", "tokenizer/bpe.codes")),
                Map.of(
                        "pre_tokenizer", "phobert_bpe",
                        "status", "metadata_only_until_phobert_bpe_runtime")));
    }
}
