package tech.kayys.aljabr.models.poolformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class PoolFormerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "poolformer",
                "PoolFormer",
                List.of("poolformer"),
                List.of("PoolFormerModel", "PoolFormerForImageClassification"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "poolformer_image_processor",
                        "direct_safetensor", "pending_poolformer_token_mixer_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
