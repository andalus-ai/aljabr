package tech.kayys.aljabr.models.speecht5;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class SpeechT5ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "speecht5",
                "SpeechT5",
                List.of("speecht5", "speecht5_hifigan"),
                List.of("SpeechT5Model", "SpeechT5ForSpeechToText", "SpeechT5ForTextToSpeech",
                        "SpeechT5ForSpeechToSpeech", "SpeechT5HifiGan"),
                List.of(
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "speecht5_processor_tokenizer",
                        "direct_safetensor", "pending_speecht5_feature_extractor_vocoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "speecht5-processor-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("vocab.json"), List.of("tokenizer/vocab.json"), List.of("tokenizer.json")),
                Map.of(
                        "processor", "processing_speecht5",
                        "feature_extractor", "feature_extraction_speecht5",
                        "vocoder", "speecht5_hifigan",
                        "status", "metadata_only")));
    }
}
