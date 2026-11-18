package tech.kayys.aljabr.models.embeddinggemma2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

/**
 * Metadata and tokenizer routing for Google's EmbeddingGemma 2 family.
 *
 * <p>This plugin intentionally does not claim direct inference readiness.
 * The embedding runner is registered separately only when its complete
 * text-forward path is implemented and validated.</p>
 */
@ApplicationScoped
public final class EmbeddingGemma2ModelFamilyPlugin implements ModelFamilyPlugin {

    public static final String FAMILY_ID = "embedding_gemma2";

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                FAMILY_ID,
                "Google EmbeddingGemma 2",
                List.of("embedding_gemma2", "embedding_gemma2_text"),
                List.of("EmbeddingGemma2Model"),
                List.of(ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TOKENIZER),
                Map.ofEntries(
                        Map.entry("bundle_profile", "optional"),
                        Map.entry("direct_safetensor", "pending_text_encoder_and_embedding_head"),
                        Map.entry("runner_status", "not_registered_until_forward_path_is_complete"),
                        Map.entry("supported_modalities", "text"),
                        Map.entry("deferred_modalities", "image,audio,video"),
                        Map.entry("text_layers", "24"),
                        Map.entry("hidden_size", "512"),
                        Map.entry("intermediate_size", "2048"),
                        Map.entry("embedding_dimension", "768"),
                        Map.entry("supported_output_dimensions", "768,512,256,128"),
                        Map.entry("max_input_tokens", "8192"),
                        Map.entry("supported_dtypes", "bfloat16,float32"),
                        Map.entry("attention", "bidirectional_mixed_full_and_sliding"),
                        Map.entry("per_layer_embeddings", "required"),
                        Map.entry("pooling", "mean"),
                        Map.entry("projection", "512_to_768"),
                        Map.entry("normalization", "l2"),
                        Map.entry("version", "0.1.0-SNAPSHOT")));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("embedding-gemma2-sentencepiece"));
    }
}
