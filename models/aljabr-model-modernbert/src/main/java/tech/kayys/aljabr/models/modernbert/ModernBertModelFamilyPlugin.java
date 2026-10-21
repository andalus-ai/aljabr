package tech.kayys.aljabr.models.modernbert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ModernBertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "modernbert",
                "ModernBERT",
                List.of("modernbert"),
                List.of("ModernBertModel", "ModernBertForMaskedLM",
                        "ModernBertForSequenceClassification", "ModernBertForTokenClassification",
                        "ModernBertForQuestionAnswering", "ModernBertForMultipleChoice"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "tokenizer_json_bpe",
                        "direct_safetensor", "not_causal_lm_modernbert_encoder_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("modernbert-hf-bpe"));
    }
}
