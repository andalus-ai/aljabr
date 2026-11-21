package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2InputRole;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2Tokenizer;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2Tokenizer.TokenizedInput;
import tech.kayys.aljabr.models.embeddinggemma2.output.EmbeddingGemma2VectorPostProcessor;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;

/**
 * Loaded text-only EmbeddingGemma 2 model: shared tokenizer adapter, canonical
 * text backbone, and the model's Matryoshka/L2 output contract.
 *
 * <p>Checkpoint deserialization is deliberately outside this model object.
 * Format adapters assemble it from validated configuration, tokenizer, and
 * checkpoint-backed weights.</p>
 */
public final class EmbeddingGemma2TextModel {
    private final EmbeddingGemma2Config config;
    private final EmbeddingGemma2Tokenizer tokenizer;
    private final EmbeddingGemma2TextBackbone backbone;

    public EmbeddingGemma2TextModel(
            EmbeddingGemma2Config config,
            EmbeddingGemma2Tokenizer tokenizer,
            EmbeddingGemma2TextBackbone backbone) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer must not be null");
        this.backbone = Objects.requireNonNull(backbone, "backbone must not be null");
        if (config.embeddingDim() != EmbeddingGemma2VectorPostProcessor.NATIVE_DIMENSION) {
            throw new IllegalArgumentException("EmbeddingGemma 2 native embedding dimension must be 768");
        }
    }

    /** Embeds an unformatted single input using the model's native 768 dimensions. */
    public float[] embed(String input) {
        return embed(input, null, EmbeddingTask.UNSPECIFIED, EmbeddingGemma2InputRole.SINGLE, null);
    }

    /**
     * Formats, tokenizes, encodes, pools, truncates, and L2-normalizes one input.
     *
     * @param input input text or code
     * @param title optional document title
     * @param task model-independent embedding task
     * @param role query, document, or symmetric single-input role
     * @param outputDimension optional Matryoshka dimension; null selects 768
     */
    public float[] embed(
            String input,
            String title,
            EmbeddingTask task,
            EmbeddingGemma2InputRole role,
            Integer outputDimension) {
        Objects.requireNonNull(input, "input must not be null");
        Objects.requireNonNull(task, "task must not be null");
        Objects.requireNonNull(role, "role must not be null");

        TokenizedInput tokens = tokenizer.tokenize(input, title, task, role);
        float[] projected = backbone.encodePooled(
                tokens.inputIds(), tokens.validTokens(), tokens.positionIds());
        int dimension = outputDimension == null
                ? EmbeddingGemma2VectorPostProcessor.NATIVE_DIMENSION
                : outputDimension;
        return EmbeddingGemma2VectorPostProcessor.process(projected, dimension);
    }

    public EmbeddingGemma2Config config() { return config; }
    public int nativeDimension() { return EmbeddingGemma2VectorPostProcessor.NATIVE_DIMENSION; }
    public int maxInputTokens() { return config.maxInputTokens(); }
}
