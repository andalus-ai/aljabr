package tech.kayys.aljabr.models.llavanextvideo;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LlavaNextVideoModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "llava_next_video",
                "LLaVA-NeXT-Video",
                List.of("llava_next_video", "llava-next-video"),
                List.of("LlavaNextVideoForConditionalGeneration", "LlavaNextVideoModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "delegated_text_backbone_bpe_or_sentencepiece",
                        "video_processor", "llava_next_video_processor",
                        "direct_safetensor", "pending_llava_next_video_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.huggingFaceBpe("llava-next-video-hf-bpe"),
                ModelTokenizerDescriptor.sentencePieceBpe("llava-next-video-spm-bpe"));
    }
}
