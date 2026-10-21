package tech.kayys.aljabr.models.esm;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class EsmModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "esm",
                "ESM / ESMFold",
                List.of("esm"),
                List.of("EsmModel", "EsmForMaskedLM", "EsmForSequenceClassification",
                        "EsmForTokenClassification", "EsmForProteinFolding"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "protein_vocab_metadata_only",
                        "direct_safetensor", "pending_protein_folding_and_contact_heads_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "esm-protein-vocab",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.txt"),
                        List.of("tokenizer/vocab.txt")),
                Map.of(
                        "pre_tokenizer", "protein_sequence_split",
                        "status", "metadata_only_until_protein_tokenizer_runtime")));
    }
}
