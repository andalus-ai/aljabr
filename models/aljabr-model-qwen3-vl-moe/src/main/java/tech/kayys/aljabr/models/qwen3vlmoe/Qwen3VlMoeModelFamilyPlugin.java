package tech.kayys.aljabr.models.qwen3vlmoe;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen3VlMoeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen3_vl_moe",
                "Alibaba Qwen3-VL MoE",
                List.of("qwen3_vl_moe", "qwen3_vl_moe_text", "qwen3_vl_moe_vision"),
                List.of("Qwen3VLMoeForConditionalGeneration", "Qwen3VLMoeModel",
                        "Qwen3VLMoeTextModel", "Qwen3VLMoeVisionModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "qwen_hf_bpe_with_qwen3_vl_moe_shared_processors",
                        "direct_safetensor", "pending_qwen3_vl_moe_sparse_expert_vision_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "qwen3-vl-moe-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_qwen3_vl_shared",
                        "video_processor", "video_processing_qwen3_vl_shared",
                        "status", "metadata_only")));
    }
}
