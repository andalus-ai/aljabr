package tech.kayys.aljabr.models.encodec;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class EncodecModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "encodec",
                "EnCodec",
                List.of("encodec"),
                List.of("EncodecModel"),
                List.of( ),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "feature_extractor", "encodec_feature_extractor",
                        "direct_safetensor", "pending_neural_audio_codec_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
