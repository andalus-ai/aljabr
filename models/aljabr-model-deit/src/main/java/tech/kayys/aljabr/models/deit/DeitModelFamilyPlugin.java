package tech.kayys.aljabr.models.deit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DeitModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "deit",
                "DeiT",
                List.of("deit"),
                List.of("DeiTModel", "DeiTForImageClassification",
                        "DeiTForImageClassificationWithTeacher", "DeiTForMaskedImageModeling"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "deit_image_processor",
                        "direct_safetensor", "pending_deit_distillation_token_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
