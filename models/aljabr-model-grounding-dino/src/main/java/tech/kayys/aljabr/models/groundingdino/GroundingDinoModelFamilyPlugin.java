package tech.kayys.aljabr.models.groundingdino;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class GroundingDinoModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "grounding_dino",
                "Grounding DINO / MM Grounding DINO",
                List.of("grounding_dino", "grounding-dino",
                        "mm_grounding_dino", "mm-grounding-dino"),
                List.of("GroundingDinoModel", "GroundingDinoForObjectDetection",
                        "MMGroundingDinoModel", "MMGroundingDinoForObjectDetection"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "wordpiece",
                        "image_processor", "grounding_dino_image_processor",
                        "direct_safetensor", "pending_multiscale_deformable_attention_detection_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("grounding-dino-wordpiece"));
    }
}
