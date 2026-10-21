package tech.kayys.aljabr.models.dinov2registers;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Dinov2WithRegistersModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "dinov2_with_registers",
                "DINOv2 With Registers",
                List.of("dinov2_with_registers", "dinov2-with-registers"),
                List.of("Dinov2WithRegistersModel", "Dinov2WithRegistersForImageClassification",
                        "Dinov2WithRegistersBackbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "vit_image_processor",
                        "direct_safetensor", "pending_dinov2_register_token_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
