package tech.kayys.aljabr.models.llavaonevision;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LlavaOnevisionModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "llava_onevision",
                "LLaVA-OneVision",
                List.of("llava_onevision", "llava-onevision"),
                List.of("LlavaOnevisionForConditionalGeneration", "LlavaOnevisionModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "delegated_qwen_or_sentencepiece_tokenizer",
                        "image_processor", "llava_onevision_image_processor",
                        "video_processor", "llava_onevision_video_processor",
                        "direct_safetensor", "pending_llava_onevision_multimodal_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.huggingFaceBpe("llava-onevision-hf-bpe"),
                ModelTokenizerDescriptor.sentencePieceBpe("llava-onevision-spm-bpe"));
    }
}
