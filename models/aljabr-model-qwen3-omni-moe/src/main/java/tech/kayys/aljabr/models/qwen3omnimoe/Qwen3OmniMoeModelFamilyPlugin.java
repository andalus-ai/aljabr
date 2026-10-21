package tech.kayys.aljabr.models.qwen3omnimoe;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen3OmniMoeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen3_omni_moe",
                "Alibaba Qwen3-Omni-MoE",
                List.of("qwen3_omni_moe", "qwen3_omni_moe_text",
                        "qwen3_omni_moe_thinker", "qwen3_omni_moe_talker",
                        "qwen3_omni_moe_vision_encoder", "qwen3_omni_moe_audio_encoder",
                        "qwen3_omni_moe_talker_text", "qwen3_omni_moe_talker_code_predictor"),
                List.of("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeThinkerTextModel",
                        "Qwen3OmniMoeThinkerForConditionalGeneration",
                        "Qwen3OmniMoeTalkerForConditionalGeneration", "Qwen3OmniMoeTalkerModel",
                        "Qwen3OmniMoeCode2WavTransformerModel",
                        "Qwen3OmniMoeTalkerCodePredictorModel",
                        "Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION, ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "qwen_hf_bpe_with_omni_moe_processors",
                        "direct_safetensor", "pending_qwen3_omni_moe_multimodal_moe_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "qwen3-omni-moe-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_qwen3_omni_moe",
                        "audio_processor", "qwen3_omni_moe_audio",
                        "video_processor", "qwen3_omni_moe_video",
                        "status", "metadata_only")));
    }
}
