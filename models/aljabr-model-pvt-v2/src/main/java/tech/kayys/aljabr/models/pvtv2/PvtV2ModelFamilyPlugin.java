package tech.kayys.aljabr.models.pvtv2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class PvtV2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "pvt_v2",
                "PVT v2",
                List.of("pvt_v2", "pvt-v2"),
                List.of("PvtV2Model", "PvtV2ForImageClassification", "PvtV2Backbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "pvt_v2_image_processor",
                        "direct_safetensor", "pending_pvt_v2_linear_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
