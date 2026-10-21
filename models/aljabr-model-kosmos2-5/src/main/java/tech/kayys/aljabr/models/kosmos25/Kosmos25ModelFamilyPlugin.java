package tech.kayys.aljabr.models.kosmos25;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Kosmos25ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "kosmos2_5",
                "Microsoft Kosmos-2.5",
                List.of("kosmos2_5", "kosmos2.5", "kosmos-2.5", "kosmos_2_5",
                        "kosmos_2_5_text_model", "kosmos_2_5_vision_model"),
                List.of("Kosmos2_5ForConditionalGeneration", "Kosmos2_5Model",
                        "Kosmos2_5TextForCausalLM", "Kosmos2_5TextModel",
                        "Kosmos2_5VisionModel", "Kosmos2_5VisionEncoder"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "processor_backed_text_grounding_tokenizer",
                        "image_processor", "kosmos2_5_image_processor",
                        "direct_safetensor", "pending_kosmos2_5_grounding_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "kosmos2-5-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_kosmos2_5",
                        "grounding", "ocr_and_layout_tokens",
                        "status", "metadata_only")));
    }
}
