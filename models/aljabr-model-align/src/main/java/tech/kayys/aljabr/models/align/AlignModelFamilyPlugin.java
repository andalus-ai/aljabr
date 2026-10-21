package tech.kayys.aljabr.models.align;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class AlignModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "align",
                "ALIGN",
                List.of("align", "align_text_model", "align_vision_model"),
                List.of("AlignModel", "AlignTextModel", "AlignVisionModel"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "wordpiece_with_align_processor",
                        "image_processor", "align_efficientnet_image_processor",
                        "direct_safetensor", "pending_align_dual_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("align-wordpiece"));
    }
}
