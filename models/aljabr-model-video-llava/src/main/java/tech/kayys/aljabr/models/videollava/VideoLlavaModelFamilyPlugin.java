package tech.kayys.aljabr.models.videollava;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class VideoLlavaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "video_llava",
                "Video-LLaVA / VIP-LLaVA",
                List.of("video_llava", "video-llava", "vipllava", "vip_llava"),
                List.of("VideoLlavaForConditionalGeneration", "VideoLlavaModel",
                        "VipLlavaForConditionalGeneration", "VipLlavaModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "llava_text_tokenizer_with_video_processor",
                        "image_processor", "video_llava_image_processor",
                        "video_processor", "video_llava_video_processor",
                        "direct_safetensor", "pending_video_frame_projector_and_clip_bridge_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "video-llava-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_video_llava",
                        "image_processor", "image_processing_video_llava",
                        "video_processor", "video_processing_video_llava",
                        "status", "metadata_only")));
    }
}
