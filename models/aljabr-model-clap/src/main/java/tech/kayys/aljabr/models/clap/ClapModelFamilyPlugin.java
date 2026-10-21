package tech.kayys.aljabr.models.clap;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ClapModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "clap",
                "CLAP",
                List.of("clap", "clap_text_model", "clap_audio_model"),
                List.of("ClapModel", "ClapTextModel", "ClapAudioModel",
                        "ClapTextModelWithProjection", "ClapAudioModelWithProjection"),
                List.of(ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "clap_processor_text_audio_features",
                        "direct_safetensor", "not_causal_lm_audio_text_embedding_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "clap-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_clap",
                        "feature_extractor", "feature_extraction_clap",
                        "status", "metadata_only")));
    }
}
