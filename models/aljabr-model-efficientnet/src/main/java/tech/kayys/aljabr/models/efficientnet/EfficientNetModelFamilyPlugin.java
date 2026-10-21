package tech.kayys.aljabr.models.efficientnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class EfficientNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "efficientnet",
                "EfficientNet",
                List.of("efficientnet"),
                List.of("EfficientNetModel", "EfficientNetForImageClassification"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "efficientnet_image_processor",
                        "direct_safetensor", "not_causal_lm_image_backbone_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
