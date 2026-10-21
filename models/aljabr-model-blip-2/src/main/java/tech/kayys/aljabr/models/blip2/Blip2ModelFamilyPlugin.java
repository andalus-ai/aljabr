package tech.kayys.aljabr.models.blip2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Blip2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "blip_2",
                "BLIP-2",
                List.of("blip_2", "blip-2", "blip2"),
                List.of("Blip2Model", "Blip2ForConditionalGeneration",
                        "Blip2QFormerModel", "Blip2ForImageTextRetrieval",
                        "Blip2VisionModel"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "processor_backed_text_tokenizer",
                        "image_processor", "blip2_image_processor",
                        "direct_safetensor", "pending_blip2_qformer_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.wordPiece("blip2-wordpiece"),
                ModelTokenizerDescriptor.huggingFaceBpe("blip2-hf-bpe"));
    }
}
