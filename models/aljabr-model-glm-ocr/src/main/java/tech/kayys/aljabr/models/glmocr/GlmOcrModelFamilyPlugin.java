package tech.kayys.aljabr.models.glmocr;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class GlmOcrModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "glm_ocr",
                "GLM-OCR",
                List.of("glm_ocr", "glm-ocr", "glm_ocr_vision", "glm_ocr_text"),
                List.of("GlmOcrModel", "GlmOcrForConditionalGeneration",
                        "GlmOcrVisionModel", "GlmOcrTextModel"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "glm_qwen_style_bpe",
                        "image_processor", "glm_ocr_image_processor",
                        "direct_safetensor", "pending_glm_ocr_vision_text_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("glm-ocr-bpe"));
    }
}
