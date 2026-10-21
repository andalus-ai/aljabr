package tech.kayys.aljabr.models.regnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RegNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "regnet",
                "RegNet",
                List.of("regnet"),
                List.of("RegNetModel", "RegNetForImageClassification"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "regnet_image_processor",
                        "direct_safetensor", "pending_regnet_stage_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
