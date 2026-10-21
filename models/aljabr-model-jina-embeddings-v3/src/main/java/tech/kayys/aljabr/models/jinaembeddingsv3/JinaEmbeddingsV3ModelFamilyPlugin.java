package tech.kayys.aljabr.models.jinaembeddingsv3;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class JinaEmbeddingsV3ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "jina_embeddings_v3",
                "Jina Embeddings v3",
                List.of("jina_embeddings_v3"),
                List.of("JinaEmbeddingsV3Model", "JinaEmbeddingsV3ForMaskedLM",
                        "JinaEmbeddingsV3ForSequenceClassification",
                        "JinaEmbeddingsV3ForTokenClassification",
                        "JinaEmbeddingsV3ForQuestionAnswering"),
                List.of( ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "xlm_roberta_sentencepiece_metadata_only",
                        "direct_safetensor", "not_causal_lm_embedding_encoder_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "jina-embeddings-v3-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(),
                Map.of(
                        "pre_tokenizer", "sentencepiece",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
