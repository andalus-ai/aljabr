package tech.kayys.aljabr.models.llavanext;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LlavaNextModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "llava_next",
                "LLaVA-NeXT",
                List.of("llava_next", "llava-next"),
                List.of("LlavaNextForConditionalGeneration", "LlavaNextModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "delegated_text_backbone_bpe_or_sentencepiece",
                        "image_processor", "llava_next_image_processor",
                        "direct_safetensor", "pending_llava_next_vision_projector_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.huggingFaceBpe("llava-next-hf-bpe"),
                ModelTokenizerDescriptor.sentencePieceBpe("llava-next-spm-bpe"));
    }
}
