package tech.kayys.aljabr.models.kosmos2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Kosmos2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "kosmos2",
                "Microsoft Kosmos-2",
                List.of("kosmos-2", "kosmos2", "kosmos_2_text_model", "kosmos_2_vision_model"),
                List.of("Kosmos2ForConditionalGeneration", "Kosmos2Model",
                        "Kosmos2TextForCausalLM", "Kosmos2TextModel", "Kosmos2VisionModel",
                        "Kosmos2VisionTransformer"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "processor_backed_text_grounding_tokenizer",
                        "image_processor", "kosmos2_image_processor",
                        "direct_safetensor", "pending_grounding_processor_and_vision_text_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "kosmos2-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_kosmos2",
                        "grounding", "phrase_and_box_tokens",
                        "status", "metadata_only")));
    }
}
