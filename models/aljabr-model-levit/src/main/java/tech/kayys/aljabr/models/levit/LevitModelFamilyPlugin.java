package tech.kayys.aljabr.models.levit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LevitModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "levit",
                "LeViT",
                List.of("levit"),
                List.of("LevitModel", "LevitForImageClassification",
                        "LevitForImageClassificationWithTeacher"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "levit_image_processor",
                        "direct_safetensor", "pending_levit_hybrid_cnn_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
