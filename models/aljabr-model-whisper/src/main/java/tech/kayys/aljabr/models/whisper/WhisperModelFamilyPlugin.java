package tech.kayys.aljabr.models.whisper;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Model-family metadata and checkpoint recognition for OpenAI Whisper.
 *
 * <p>This plugin deliberately declares audio/model-family support without
 * claiming direct Safetensors inference readiness. The encoder-decoder
 * execution path must pass its own conformance suite before that claim is
 * added to the descriptor.</p>
 */
@ApplicationScoped
public final class WhisperModelFamilyPlugin implements ModelFamilyPlugin {

    public static final String FAMILY_ID = "whisper";

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                FAMILY_ID,
                "OpenAI Whisper",
                List.of("whisper"),
                List.of("WhisperForConditionalGeneration", "WhisperModel",
                        "WhisperEncoder", "WhisperDecoder"),
                List.of(ModelFamilyCapability.AUDIO, ModelFamilyCapability.TOKENIZER),
                Map.ofEntries(
                        Map.entry("bundle_profile", "optional"),
                        Map.entry("task_family", "automatic_speech_recognition"),
                        Map.entry("supported_tasks", "transcription,translation"),
                        Map.entry("audio_input", "mono_pcm"),
                        Map.entry("audio_sample_rate", "16000"),
                        Map.entry("audio_chunk_seconds", "30"),
                        Map.entry("audio_feature_extractor", "WhisperFeatureExtractor"),
                        Map.entry("audio_feature_bins", "model_config_dependent"),
                        Map.entry("direct_safetensor", "pending_encoder_decoder_audio_runtime"),
                        Map.entry("large_v3_model_id", "openai/whisper-large-v3"),
                        Map.entry("large_v3_mel_bins", "128"),
                        Map.entry("version", "0.1.0-SNAPSHOT")));
    }

    @Override
    public boolean matchesCheckpoint(Path modelDirectory) {
        return WhisperCheckpointInspector.isWhisperCheckpoint(modelDirectory);
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("whisper-byte-level-bpe"));
    }
}
