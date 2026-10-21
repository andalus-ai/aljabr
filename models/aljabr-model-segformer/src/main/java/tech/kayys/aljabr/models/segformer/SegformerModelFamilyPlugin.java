package tech.kayys.aljabr.models.segformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class SegformerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "segformer",
                "SegFormer",
                List.of("segformer"),
                List.of("SegformerModel", "SegformerForImageClassification",
                        "SegformerForSemanticSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "segformer_image_processor",
                        "direct_safetensor", "pending_segformer_decode_head_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
