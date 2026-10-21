package tech.kayys.aljabr.models.layoutlm;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LayoutLmModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "layoutlm",
                "LayoutLM / LayoutLMv2 / LayoutLMv3",
                List.of("layoutlm", "layoutlmv2", "layoutlmv3"),
                List.of("LayoutLMModel", "LayoutLMForMaskedLM", "LayoutLMForTokenClassification",
                        "LayoutLMv2Model", "LayoutLMv2ForTokenClassification",
                        "LayoutLMv3Model", "LayoutLMv3ForTokenClassification"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "layout_aware_wordpiece_or_bpe_processor",
                        "direct_safetensor", "not_causal_lm_layout_processor_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.wordPiece("layoutlm-wordpiece"),
                new ModelTokenizerDescriptor(
                        "layoutlm-processor-tokenizer-json",
                        ModelTokenizerKind.CUSTOM,
                        List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                        Map.of(
                                "processor", "processing_layoutlmv2_or_layoutlmv3",
                                "image_processor", "layoutlm_image_processor",
                                "requires", "bbox_or_ocr_layout",
                                "status", "metadata_only")));
    }
}
