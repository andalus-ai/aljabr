package tech.kayys.aljabr.models.visualbert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class VisualBertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "visual_bert",
                "VisualBERT",
                List.of("visual_bert", "visual-bert"),
                List.of("VisualBertModel", "VisualBertForPreTraining",
                        "VisualBertForQuestionAnswering", "VisualBertForMultipleChoice",
                        "VisualBertForVisualReasoning", "VisualBertForRegionToPhraseAlignment"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "wordpiece",
                        "direct_safetensor", "pending_visual_feature_fusion_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("visual-bert-wordpiece"));
    }
}
