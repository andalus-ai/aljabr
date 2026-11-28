package tech.kayys.aljabr.models.qwen3asr;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;
import java.util.List;
import java.util.Map;

/**
 * Checkpoint-family metadata for Qwen3-ASR. Inference readiness remains pending
 * until the acoustic encoder, projector, decoder integration and tokenizer prompt
 * path pass checkpoint-backed conformance tests.
 */
@ApplicationScoped
public final class Qwen3AsrModelFamilyPlugin implements ModelFamilyPlugin {
    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen3_asr", "Qwen3-ASR",
                List.of("qwen3_asr", "qwen3_asr_encoder"),
                List.of("Qwen3ASRForConditionalGeneration", "Qwen3ASRAudioEncoder"),
                List.of(ModelFamilyCapability.AUDIO, ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.MULTIMODAL),
                Map.ofEntries(
                        Map.entry("bundle_profile", "optional"),
                        Map.entry("task_family", "automatic_speech_recognition"),
                        Map.entry("supported_checkpoints", "Qwen/Qwen3-ASR-1.7B-hf"),
                        Map.entry("audio_frontend", "qwen3_asr_whisper_style"),
                        Map.entry("audio_sample_rate", "16000"),
                        Map.entry("audio_feature_bins", "128"),
                        Map.entry("audio_token_id", "151676"),
                        Map.entry("timestamp_token_id", "151705"),
                        Map.entry("text_backbone", "qwen3"),
                        Map.entry("direct_safetensor", "pending_acoustic_encoder_projector_and_decoder_runtime"),
                        Map.entry("codec_provider", "external_optional_suling"),
                        Map.entry("version", "0.1.0-SNAPSHOT")));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "qwen3-asr-qwen-tokenizer", ModelTokenizerKind.HUGGING_FACE_BPE,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of("tokenizer_family", "qwen3", "audio_token_id", "151676",
                        "timestamp_token_id", "151705", "status", "checkpoint_metadata_only")));
    }
}
