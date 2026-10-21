package tech.kayys.aljabr.models.qwen25omni;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen25OmniModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen2_5_omni",
                "Alibaba Qwen2.5-Omni",
                List.of("qwen2_5_omni", "qwen2_5_omni_text", "qwen2_5_omni_thinker",
                        "qwen2_5_omni_talker", "qwen2_5_omni_vision_encoder",
                        "qwen2_5_omni_audio_encoder", "qwen2_5_omni_token2wav",
                        "qwen2_5_omni_dit", "qwen2_5_omni_bigvgan"),
                List.of("Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniThinkerTextModel",
                        "Qwen2_5OmniThinkerForConditionalGeneration", "Qwen2_5OmniTalkerModel",
                        "Qwen2_5OmniTalkerForConditionalGeneration",
                        "Qwen2_5OmniToken2WavModel", "Qwen2_5OmniToken2WavDiTModel",
                        "Qwen2_5OmniToken2WavBigVGANModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION, ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "qwen_hf_bpe_with_omni_processors",
                        "direct_safetensor", "pending_qwen2_5_omni_thinker_talker_token2wav_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "qwen2-5-omni-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_qwen2_5_omni",
                        "audio_processor", "qwen2_5_omni_audio",
                        "video_processor", "qwen2_5_omni_video",
                        "status", "metadata_only")));
    }
}
