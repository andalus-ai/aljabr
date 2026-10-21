package tech.kayys.aljabr.models.data2vec;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Data2VecModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "data2vec",
                "Data2Vec Text / Audio / Vision",
                List.of("data2vec-text", "data2vec_text",
                        "data2vec-audio", "data2vec_audio",
                        "data2vec-vision", "data2vec_vision"),
                List.of("Data2VecTextModel", "Data2VecTextForMaskedLM",
                        "Data2VecTextForSequenceClassification",
                        "Data2VecAudioModel", "Data2VecAudioForCTC",
                        "Data2VecAudioForSequenceClassification",
                        "Data2VecVisionModel", "Data2VecVisionForImageClassification",
                        "Data2VecVisionForSemanticSegmentation"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.VISION),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "roberta_byte_level_bpe_for_text",
                        "audio_processor", "wav2vec2_style_feature_extractor",
                        "image_processor", "beit_style_image_processor",
                        "direct_safetensor", "pending_multimodal_data2vec_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("data2vec-text-byte-level-bpe"));
    }
}
