package tech.kayys.aljabr.models.fsmt;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class FsmtModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "fsmt",
                "FairSeq Machine Translation",
                List.of("fsmt"),
                List.of("FSMTForConditionalGeneration", "FSMTModel"),
                List.of(
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "moses_bpe",
                        "direct_safetensor", "pending_fsmt_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "fsmt-moses-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("vocab-src.json", "merges.txt"),
                        List.of("vocab-tgt.json", "merges.txt"),
                        List.of("tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_fsmt",
                        "pre_tokenizer", "moses",
                        "status", "metadata_only")));
    }
}
