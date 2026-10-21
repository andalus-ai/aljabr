package tech.kayys.aljabr.models.bertweet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BertweetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bertweet",
                "BERTweet",
                List.of("bertweet"),
                List.of("RobertaModel", "RobertaForMaskedLM",
                        "RobertaForSequenceClassification", "RobertaForTokenClassification"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "bertweet_bpe_metadata_only",
                        "direct_safetensor", "roberta_layout_tokenizer_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bertweet-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.txt", "bpe.codes"),
                        List.of("tokenizer/vocab.txt", "tokenizer/bpe.codes")),
                Map.of(
                        "pre_tokenizer", "bertweet_normalizer_bpe",
                        "status", "metadata_only_until_tweet_normalizer_runtime")));
    }
}
