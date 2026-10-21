package tech.kayys.aljabr.models.byt5;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ByT5ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "byt5",
                "ByT5",
                List.of("byt5"),
                List.of("ByT5Tokenizer", "ByT5TokenizerFast"),
                List.of(
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "byte_level_no_vocab",
                        "direct_safetensor", "not_direct_runtime_tokenizer_only_over_t5_backbone",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "byt5-byte-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json"),
                        List.of("tokenizer_config.json"), List.of("tokenizer/tokenizer_config.json")),
                Map.of(
                        "pre_tokenizer", "raw_bytes",
                        "status", "metadata_only_until_byt5_tokenizer_runtime")));
    }
}
