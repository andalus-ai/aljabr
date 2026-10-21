package tech.kayys.aljabr.models.wav2vec2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Wav2Vec2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "wav2vec2",
                "Wav2Vec2 / Wav2Vec2-Conformer",
                List.of("wav2vec2", "wav2vec2_conformer", "wav2vec2-conformer"),
                List.of("Wav2Vec2Model", "Wav2Vec2ForCTC", "Wav2Vec2ForSequenceClassification",
                        "Wav2Vec2ForAudioFrameClassification", "Wav2Vec2ConformerModel",
                        "Wav2Vec2ConformerForCTC"),
                List.of( ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "feature_extractor", "wav2vec2_feature_extractor",
                        "tokenizer", "ctc_vocab_metadata_only",
                        "direct_safetensor", "pending_audio_feature_extractor_and_ctc_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "wav2vec2-ctc-vocab",
                ModelTokenizerKind.CUSTOM,
                List.of(),
                Map.of(
                        "decoder", "ctc",
                        "status", "metadata_only_until_ctc_tokenizer_runtime")));
    }
}
