package tech.kayys.aljabr.models.mobilevitv2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MobileVitV2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "mobilevitv2",
                "MobileViT v2",
                List.of("mobilevitv2", "mobilevit_v2", "mobilevit-v2"),
                List.of("MobileViTV2Model", "MobileViTV2ForImageClassification",
                        "MobileViTV2ForSemanticSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "mobilevitv2_image_processor",
                        "direct_safetensor", "pending_mobilevitv2_separable_self_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
