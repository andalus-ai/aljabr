package tech.kayys.aljabr.models.detr;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DetrModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "detr",
                "DETR / Deformable DETR",
                List.of("detr", "conditional_detr", "conditional-detr",
                        "deformable_detr", "deformable-detr", "rt_detr", "rt-detr"),
                List.of("DetrModel", "DetrForObjectDetection", "DetrForSegmentation",
                        "ConditionalDetrModel", "ConditionalDetrForObjectDetection",
                        "DeformableDetrModel", "DeformableDetrForObjectDetection",
                        "RTDetrModel", "RTDetrForObjectDetection"),
                List.of(ModelFamilyCapability.VISION),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "detr_image_processor",
                        "direct_safetensor", "not_causal_lm",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
