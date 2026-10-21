package tech.kayys.aljabr.models.qwen2vl;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen2VlModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen2_vl",
                "Alibaba Qwen2-VL",
                List.of("qwen2_vl", "qwen2_vl_text", "qwen2_vl_vision"),
                List.of("Qwen2VLForConditionalGeneration", "Qwen2VLModel", "Qwen2VLTextModel",
                        "Qwen2VisionTransformerPretrainedModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "qwen_hf_bpe_with_qwen2_vl_processors",
                        "direct_safetensor", "pending_qwen2_vl_vision_projector_mrope_and_video_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "qwen2-vl-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_qwen2_vl",
                        "image_processor", "image_processing_qwen2_vl",
                        "video_processor", "video_processing_qwen2_vl",
                        "status", "metadata_only")));
    }
}
