package tech.kayys.aljabr.models.cvt;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class CvtModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "cvt",
                "CvT",
                List.of("cvt"),
                List.of("CvtModel", "CvtForImageClassification"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "cvt_image_processor",
                        "direct_safetensor", "pending_cvt_convolutional_projection_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
