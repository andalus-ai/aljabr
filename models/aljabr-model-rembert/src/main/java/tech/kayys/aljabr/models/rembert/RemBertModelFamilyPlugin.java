package tech.kayys.aljabr.models.rembert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RemBertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "rembert",
                "RemBERT",
                List.of("rembert"),
                List.of("RemBertModel", "RemBertForMaskedLM", "RemBertForCausalLM",
                        "RemBertForSequenceClassification", "RemBertForTokenClassification",
                        "RemBertForQuestionAnswering"),
                List.of( ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_rembert_unigram_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "rembert-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("sentencepiece.model"),
                        List.of("tokenizer/tokenizer.json"),
                        List.of("tokenizer/sentencepiece.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_unigram",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
