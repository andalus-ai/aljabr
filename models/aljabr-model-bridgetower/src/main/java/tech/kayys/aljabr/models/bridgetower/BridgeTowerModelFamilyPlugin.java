package tech.kayys.aljabr.models.bridgetower;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BridgeTowerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bridgetower",
                "BridgeTower",
                List.of("bridgetower", "bridgetower_text_model", "bridgetower_vision_model"),
                List.of("BridgeTowerModel", "BridgeTowerTextModel", "BridgeTowerVisionModel",
                        "BridgeTowerForMaskedLM", "BridgeTowerForImageAndTextRetrieval",
                        "BridgeTowerForContrastiveLearning"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "processor_backed_wordpiece",
                        "image_processor", "bridgetower_image_processor",
                        "direct_safetensor", "pending_bridgetower_cross_modal_layer_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bridgetower-processor-wordpiece",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("vocab.txt"), List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_bridgetower",
                        "image_processor", "image_processing_bridgetower",
                        "tokenizer", "wordpiece",
                        "status", "metadata_only")));
    }
}
