package tech.kayys.aljabr.models.dinov2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Dinov2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "dinov2",
                "DINOv2",
                List.of("dinov2"),
                List.of("Dinov2Model", "Dinov2ForImageClassification", "Dinov2Backbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "vit_image_processor",
                        "direct_safetensor", "pending_dinov2_backbone_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
