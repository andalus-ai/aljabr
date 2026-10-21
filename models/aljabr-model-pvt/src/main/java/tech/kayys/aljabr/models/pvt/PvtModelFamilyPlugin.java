package tech.kayys.aljabr.models.pvt;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class PvtModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "pvt",
                "PVT",
                List.of("pvt"),
                List.of("PvtModel", "PvtForImageClassification"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "pvt_image_processor",
                        "direct_safetensor", "pending_pvt_spatial_reduction_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
