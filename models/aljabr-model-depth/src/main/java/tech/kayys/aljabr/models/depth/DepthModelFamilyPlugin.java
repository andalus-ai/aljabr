package tech.kayys.aljabr.models.depth;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DepthModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "depth",
                "DPT / Depth Anything / DepthPro / ZoeDepth",
                List.of("dpt", "depth_anything", "depth-anything", "depth_pro",
                        "depth-pro", "zoedepth", "zoe_depth"),
                List.of("DPTModel", "DPTForDepthEstimation", "DPTForSemanticSegmentation",
                        "DepthAnythingForDepthEstimation", "DepthProModel",
                        "DepthProForDepthEstimation", "ZoeDepthForDepthEstimation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "depth_estimation_image_processor",
                        "direct_safetensor", "pending_depth_decoder_metric_head_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
