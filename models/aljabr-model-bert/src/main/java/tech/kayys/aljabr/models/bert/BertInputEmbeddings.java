package tech.kayys.aljabr.models.bert;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.Embedding;
import tech.kayys.aljabr.core.nn.LayerNorm;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Shared BERT-family input embedding composition.
 *
 * <p>Combines word, absolute position, and token-type embeddings followed by
 * the checkpoint-configured LayerNorm. Inputs and outputs use batch-first
 * layouts: token IDs are {@code [batch, sequence]} and hidden states are
 * {@code [batch, sequence, hiddenSize]}.</p>
 *
 * <p>Embedding tables are initialized by the normal module initializer and
 * are intended to be populated from the model checkpoint before inference.</p>
 */
public final class BertInputEmbeddings extends NNModule {
    private final BertConfig config;
    private final Embedding wordEmbeddings;
    private final Embedding positionEmbeddings;
    private final Embedding tokenTypeEmbeddings;
    private final LayerNorm layerNorm;

    public BertInputEmbeddings(ComputeBackend backend, BertConfig config) {
        super(Objects.requireNonNull(backend, "backend must not be null"));
        this.config = Objects.requireNonNull(config, "config must not be null");
        wordEmbeddings = registerModule("word_embeddings",
                new Embedding(backend, config.vocabularySize(), config.hiddenSize(), config.paddingTokenId()));
        positionEmbeddings = registerModule("position_embeddings",
                new Embedding(backend, config.maxPositionEmbeddings(), config.hiddenSize()));
        tokenTypeEmbeddings = registerModule("token_type_embeddings",
                new Embedding(backend, config.typeVocabularySize(), config.hiddenSize()));
        layerNorm = registerModule("layer_norm",
                new LayerNorm(backend, new long[] {config.hiddenSize()},
                        (float) config.layerNormEpsilon(), true));
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        return forward(inputIds, null);
    }

    /**
     * @param inputIds I32 token IDs {@code [batch, sequence]}
     * @param tokenTypeIds optional I32 segment IDs with the same shape; defaults to zeros
     */
    public Tensor forward(Tensor inputIds, Tensor tokenTypeIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        validateInputBackend(inputIds);
        if (inputIds.dtype() != DType.I32 || inputIds.shape().rank() != 2) {
            throw new IllegalArgumentException("inputIds must be an I32 tensor with shape [batch, sequence]");
        }
        long batch = inputIds.shape().dim(0);
        long sequence = inputIds.shape().dim(1);
        if (batch <= 0 || sequence <= 0 || sequence > config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("input batch and sequence must be positive and sequence must not exceed "
                    + config.maxPositionEmbeddings());
        }
        if (tokenTypeIds != null) {
            validateInputBackend(tokenTypeIds);
            if (tokenTypeIds.dtype() != DType.I32
                    || !java.util.Arrays.equals(tokenTypeIds.shape().dims(), inputIds.shape().dims())) {
                throw new IllegalArgumentException("tokenTypeIds must be I32 and match inputIds shape");
            }
        }

        Tensor generatedPositions = null;
        Tensor generatedTokenTypes = null;
        Tensor word = null;
        Tensor position = null;
        Tensor tokenType = null;
        Tensor sum = null;
        Tensor withPositions = null;
        try {
            Tensor positions;
            if (sequence > Integer.MAX_VALUE || batch > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("input dimensions exceed supported host index range");
            }
            int count = Math.toIntExact(Math.multiplyExact(batch, sequence));
            int[] positionValues = new int[count];
            for (int b = 0; b < batch; b++) {
                for (int s = 0; s < sequence; s++) {
                    positionValues[Math.toIntExact(b * sequence + s)] = s;
                }
            }
            generatedPositions = TensorFactory.of(positionValues, inputIds.backend(), batch, sequence);
            Tensor effectiveTokenTypes = tokenTypeIds;
            if (effectiveTokenTypes == null) {
                generatedTokenTypes = TensorFactory.of(new int[count], inputIds.backend(), batch, sequence);
                effectiveTokenTypes = generatedTokenTypes;
            }

            word = wordEmbeddings.forward(inputIds);
            position = positionEmbeddings.forward(generatedPositions);
            tokenType = tokenTypeEmbeddings.forward(effectiveTokenTypes);
            sum = word.add(position);
            withPositions = sum.add(tokenType);
            return layerNorm.forward(withPositions);
        } finally {
            if (withPositions != null) withPositions.release();
            if (sum != null) sum.release();
            if (tokenType != null) tokenType.release();
            if (position != null) position.release();
            if (word != null) word.release();
            if (generatedTokenTypes != null) generatedTokenTypes.release();
            if (generatedPositions != null) generatedPositions.release();
        }
    }

    public BertConfig config() {
        return config;
    }
}
