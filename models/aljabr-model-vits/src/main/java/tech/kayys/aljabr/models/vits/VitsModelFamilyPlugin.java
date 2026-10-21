package tech.kayys.aljabr.models.vits;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class VitsModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "vits",
                "VITS",
                List.of("vits"),
                List.of("VitsModel"),
                List.of(
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "vits_text_tokenizer",
                        "direct_safetensor", "pending_vits_duration_flow_decoder_vocoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "vits-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("vocab.json"), List.of("tokenizer/vocab.json"), List.of("tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_vits",
                        "status", "metadata_only")));
    }
}
