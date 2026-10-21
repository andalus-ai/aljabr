package tech.kayys.aljabr.models.bigbird;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BigBirdModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bigbird",
                "Google BigBird",
                List.of("big_bird", "bigbird"),
                List.of("BigBirdModel", "BigBirdForMaskedLM", "BigBirdForCausalLM",
                        "BigBirdForPreTraining", "BigBirdForSequenceClassification",
                        "BigBirdForQuestionAnswering", "BigBirdForTokenClassification"),
                List.of( ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "wordpiece",
                        "direct_safetensor", "pending_bigbird_block_sparse_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("bigbird-wordpiece"));
    }
}
