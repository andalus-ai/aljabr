package tech.kayys.aljabr.models.yolos;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class YolosModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "yolos",
                "YOLOS",
                List.of("yolos"),
                List.of("YolosModel", "YolosForObjectDetection"),
                List.of(ModelFamilyCapability.VISION),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "yolos_image_processor",
                        "direct_safetensor", "pending_yolos_detection_head_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
