package tech.kayys.aljabr.models.sam;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class SamModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "sam",
                "Segment Anything",
                List.of("sam", "sam_hq", "sam-hq", "sam2", "segment_anything"),
                List.of("SamModel", "SamHQModel", "Sam2Model"),
                List.of(ModelFamilyCapability.VISION),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "image_processor", "sam_image_processor",
                        "direct_safetensor", "not_causal_lm",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
