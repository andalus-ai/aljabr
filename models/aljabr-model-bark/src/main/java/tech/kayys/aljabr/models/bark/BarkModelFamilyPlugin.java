package tech.kayys.aljabr.models.bark;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BarkModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bark",
                "Bark",
                List.of("bark"),
                List.of("BarkModel", "BarkSemanticModel", "BarkCoarseModel",
                        "BarkFineModel", "BarkCausalModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "processor_backed_text_to_audio_tokenizer",
                        "direct_safetensor", "pending_bark_semantic_coarse_fine_codec_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bark-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_bark",
                        "codec", "encodec",
                        "status", "metadata_only")));
    }
}
