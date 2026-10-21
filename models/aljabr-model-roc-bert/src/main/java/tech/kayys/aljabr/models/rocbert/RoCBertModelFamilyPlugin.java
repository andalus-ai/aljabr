package tech.kayys.aljabr.models.rocbert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RoCBertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "roc_bert",
                "RoCBERT",
                List.of("roc_bert", "roc-bert"),
                List.of("RoCBertForPreTraining", "RoCBertForMaskedLM", "RoCBertForCausalLM",
                        "RoCBertForSequenceClassification", "RoCBertForMultipleChoice",
                        "RoCBertForTokenClassification", "RoCBertForQuestionAnswering",
                        "RoCBertTokenizer"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "rocbert_wordpiece_with_shape_pronunciation",
                        "direct_safetensor", "pending_rocbert_multifeature_embeddings",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("roc-bert-wordpiece"));
    }
}
