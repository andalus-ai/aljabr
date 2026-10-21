package tech.kayys.aljabr.models.mobilevit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MobileVitModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "mobilevit",
                "MobileViT",
                List.of("mobilevit"),
                List.of("MobileViTModel", "MobileViTForImageClassification",
                        "MobileViTForSemanticSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "mobilevit_image_processor",
                        "direct_safetensor", "pending_mobilevit_hybrid_conv_transformer_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
