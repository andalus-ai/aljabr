package tech.kayys.aljabr.models.mgpstr;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MgpStrModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "mgp_str",
                "MGP-STR",
                List.of("mgp-str", "mgp_str"),
                List.of("MgpstrModel", "MgpstrForSceneTextRecognition"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "character_vocab_metadata_only",
                        "processor", "mgp_str_scene_text_processor",
                        "direct_safetensor", "pending_scene_text_decoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "mgp-str-char-vocab",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.json"),
                        List.of("tokenizer/vocab.json")),
                Map.of(
                        "pre_tokenizer", "character",
                        "status", "metadata_only_until_scene_text_tokenizer_runtime")));
    }
}
