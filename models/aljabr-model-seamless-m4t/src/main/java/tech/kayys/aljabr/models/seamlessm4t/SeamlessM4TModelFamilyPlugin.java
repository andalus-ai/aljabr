package tech.kayys.aljabr.models.seamlessm4t;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class SeamlessM4TModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "seamless_m4t",
                "SeamlessM4T / SeamlessM4Tv2",
                List.of("seamless_m4t", "seamless_m4t_v2"),
                List.of("SeamlessM4TModel", "SeamlessM4TForTextToText",
                        "SeamlessM4TForSpeechToText", "SeamlessM4TForTextToSpeech",
                        "SeamlessM4TForSpeechToSpeech", "SeamlessM4Tv2Model",
                        "SeamlessM4Tv2ForTextToText", "SeamlessM4Tv2ForSpeechToText",
                        "SeamlessM4Tv2ForTextToSpeech", "SeamlessM4Tv2ForSpeechToSpeech"),
                List.of(
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "seamless_m4t_processor_tokenizer",
                        "direct_safetensor", "pending_seamless_m4t_conformer_t2u_hifigan_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "seamless-m4t-processor-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_seamless_m4t",
                        "feature_extractor", "feature_extraction_seamless_m4t",
                        "tokenizer", "tokenization_seamless_m4t",
                        "status", "metadata_only")));
    }
}
