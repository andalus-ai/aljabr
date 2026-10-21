package tech.kayys.aljabr.models.flux;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

/**
 * Model family descriptor plugin for Black Forest Labs FLUX.1 and FLUX.2 architectures.
 */
@ApplicationScoped
public class FluxModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "flux",
                "Black Forest Labs FLUX",
                List.of("flux", "flux.1-schnell", "flux.1-dev", "flux.2-klein-9b", "flux_transformer_2d"),
                List.of("FluxTransformer2DModel", "FluxModel", "FluxPipeline", "Flux2Klein"),
                List.of(
                        ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION,
                        ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TOKENIZER
                ),
                Map.of(
                        "bundle_profile", "flux_dit",
                        "origin", "black-forest-labs/FLUX.1-schnell",
                        "text_encoder_1", "clip-l",
                        "text_encoder_2", "t5-xxl",
                        "latent_channels", "16",
                        "vae_scale_factor", "0.3611",
                        "scheduler", "flow_matching_euler",
                        "version", "0.1.0-SNAPSHOT"
                )
        );
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.huggingFaceBpe("clip-l"),
                ModelTokenizerDescriptor.sentencePieceBpe("t5-xxl-spm")
        );
    }
}
