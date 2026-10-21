package tech.kayys.aljabr.models.vipllava;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class VipLlavaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "vipllava",
                "Vip-LLaVA",
                List.of("vipllava", "vip_llava", "vip-llava"),
                List.of("VipLlavaForConditionalGeneration", "VipLlavaModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "delegated_llama_or_clip_tokenizer",
                        "image_processor", "vipllava_image_processor",
                        "direct_safetensor", "pending_vipllava_visual_prompt_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.huggingFaceBpe("vipllava-hf-bpe"),
                ModelTokenizerDescriptor.sentencePieceBpe("vipllava-spm-bpe"));
    }
}
