package tech.kayys.aljabr.models.qwen25vl;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen25VlModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen2_5_vl",
                "Alibaba Qwen2.5-VL",
                List.of("qwen2_5_vl", "qwen2_5_vl_text", "qwen2_5_vl_vision"),
                List.of("Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel",
                        "Qwen2_5_VLTextModel", "Qwen2_5_VisionTransformerPretrainedModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "qwen_hf_bpe_with_qwen2_5_vl_processor",
                        "direct_safetensor", "pending_qwen2_5_vl_windowed_vision_mrope_and_video_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "qwen2-5-vl-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_qwen2_5_vl",
                        "vision_processor", "shared_qwen2_vl_image_video_processors",
                        "status", "metadata_only")));
    }
}
