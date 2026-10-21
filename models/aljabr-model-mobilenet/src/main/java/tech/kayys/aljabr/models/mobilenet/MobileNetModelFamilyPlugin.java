package tech.kayys.aljabr.models.mobilenet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MobileNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "mobilenet",
                "MobileNet V1 / V2",
                List.of("mobilenet_v1", "mobilenet-v1", "mobilenet_v2", "mobilenet-v2"),
                List.of("MobileNetV1Model", "MobileNetV1ForImageClassification",
                        "MobileNetV2Model", "MobileNetV2ForImageClassification",
                        "MobileNetV2ForSemanticSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "mobilenet_image_processor",
                        "direct_safetensor", "pending_depthwise_conv_image_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
