package tech.kayys.aljabr.models.beit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BeitModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "beit",
                "BEiT",
                List.of("beit"),
                List.of("BeitModel", "BeitForMaskedImageModeling",
                        "BeitForImageClassification", "BeitForSemanticSegmentation",
                        "BeitBackbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "beit_image_processor",
                        "direct_safetensor", "pending_beit_relative_position_and_segmentation_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
