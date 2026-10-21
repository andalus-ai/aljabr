package tech.kayys.aljabr.models.robertaprelayernorm;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RobertaPreLayerNormModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "roberta_prelayernorm",
                "RoBERTa PreLayerNorm",
                List.of("roberta_prelayernorm", "roberta-prelayernorm",
                        "roberta_pre_layer_norm", "roberta-pre-layer-norm"),
                List.of("RobertaPreLayerNormModel", "RobertaPreLayerNormForCausalLM",
                        "RobertaPreLayerNormForMaskedLM",
                        "RobertaPreLayerNormForSequenceClassification",
                        "RobertaPreLayerNormForTokenClassification",
                        "RobertaPreLayerNormForQuestionAnswering"),
                List.of( ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "byte_level_bpe",
                        "direct_safetensor", "pending_roberta_prelayernorm_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("roberta-prelayernorm-byte-level-bpe"));
    }
}
