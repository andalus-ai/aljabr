package tech.kayys.aljabr.models.focalnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class FocalNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "focalnet",
                "FocalNet",
                List.of("focalnet"),
                List.of("FocalNetModel", "FocalNetForImageClassification",
                        "FocalNetForMaskedImageModeling", "FocalNetBackbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "focalnet_image_processor",
                        "direct_safetensor", "pending_focal_modulation_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
