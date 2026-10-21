package tech.kayys.aljabr.models.chineseclip;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ChineseClipModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "chinese_clip",
                "ChineseCLIP",
                List.of("chinese_clip", "chinese_clip_text_model", "chinese_clip_vision_model"),
                List.of("ChineseCLIPModel", "ChineseCLIPTextModel", "ChineseCLIPVisionModel"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "chinese_wordpiece_with_clip_processor",
                        "image_processor", "chinese_clip_image_processor",
                        "direct_safetensor", "pending_chinese_clip_dual_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("chinese-clip-wordpiece"));
    }
}
