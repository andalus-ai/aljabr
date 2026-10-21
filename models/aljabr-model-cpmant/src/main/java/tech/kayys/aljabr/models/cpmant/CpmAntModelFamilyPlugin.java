package tech.kayys.aljabr.models.cpmant;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class CpmAntModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "cpmant",
                "CPM-Ant",
                List.of("cpmant", "cpm-ant"),
                List.of("CpmAntForCausalLM", "CpmAntModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "cpmant_wordpiece_with_rjieba",
                        "direct_safetensor", "pending_cpmant_segment_position_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "cpmant-wordpiece",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("vocab.txt"), List.of("tokenizer/vocab.txt"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "pre_tokenizer", "rjieba",
                        "subword_tokenizer", "wordpiece",
                        "status", "metadata_only_until_cpmant_tokenizer_runtime")));
    }
}
