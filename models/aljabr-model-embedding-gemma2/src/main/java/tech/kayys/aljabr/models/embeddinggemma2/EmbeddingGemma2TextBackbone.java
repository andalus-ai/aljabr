package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2AttentionBlock;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2EncoderLayer;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2SequenceRmsNorm;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfig;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfigResolver;
import tech.kayys.aljabr.models.embeddinggemma2.output.EmbeddingGemma2MeanPooler;
import tech.kayys.aljabr.models.embeddinggemma2.ple.EmbeddingGemma2PerLayerEmbeddingProjector;

/**
 * Canonical float32 reference forward path for the EmbeddingGemma 2 text tower.
 *
 * <p>The backbone owns vocabulary lookup and embedding scaling, projection-only
 * PLE generation, construction and execution of every configured encoder
 * layer, final RMSNorm, and token-wise embedding projection. Mean pooling is
 * exposed as a separate convenience operation so callers can also inspect
 * token-level projected states.</p>
 *
 * <p>All trainable tensors are supplied explicitly in {@link Weights}. This
 * class does not deserialize checkpoint files, tokenize text, or implement
 * the image/audio/video towers.</p>
 */
public final class EmbeddingGemma2TextBackbone {
    private final EmbeddingGemma2Config config;
    private final Weights weights;
    private final List<EmbeddingGemma2EncoderLayer> layers;

    public EmbeddingGemma2TextBackbone(EmbeddingGemma2Config config, Weights weights) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        validateWeights();

        List<EmbeddingGemma2EncoderLayer> built = new ArrayList<>(config.numHiddenLayers());
        for (int index = 0; index < config.numHiddenLayers(); index++) {
            LayerWeights layerWeights = Objects.requireNonNull(weights.layers().get(index),
                    "layer weights must not be null at index " + index);
            EmbeddingGemma2LayerConfig layerConfig =
                    EmbeddingGemma2LayerConfigResolver.resolve(config, index);
            built.add(new EmbeddingGemma2EncoderLayer(config, layerConfig,
                    layerWeights.attention(), layerWeights.mlp(), layerWeights.encoder()));
        }
        this.layers = List.copyOf(built);
    }

    /**
     * Executes token IDs through the complete text tower.
     *
     * @param inputIds token IDs [sequence]
     * @param validTokens optional true=non-padding mask; null means all valid
     * @param positionIds explicit position IDs [sequence]
     * @return learned per-token projected states [sequence][embeddingDim]
     */
    public float[][] encode(int[] inputIds, boolean[] validTokens, int[] positionIds) {
        validateInputs(inputIds, validTokens, positionIds);
        float[][] embeddings = embed(inputIds);
        float[][][] perLayerInputs = EmbeddingGemma2PerLayerEmbeddingProjector.project(
                embeddings, weights.pleProjection(), weights.pleNormScale(),
                config.numHiddenLayers(), config.hiddenSizePerLayerInput(),
                (float) config.rmsNormEps());

        float[][] hidden = copy(embeddings);
        for (int layer = 0; layer < layers.size(); layer++) {
            float[][] layerPle = new float[inputIds.length][config.hiddenSizePerLayerInput()];
            for (int token = 0; token < inputIds.length; token++) {
                System.arraycopy(perLayerInputs[token][layer], 0, layerPle[token], 0,
                        config.hiddenSizePerLayerInput());
            }
            hidden = layers.get(layer).compute(hidden, layerPle, positionIds, validTokens);
        }

        float[][] normalized = EmbeddingGemma2SequenceRmsNorm.apply(
                hidden, weights.finalNormScale(), (float) config.rmsNormEps());
        return EmbeddingGemma2LinearProjection.apply(
                normalized, weights.embeddingProjection(), null);
    }

    /** Convenience path for one unpadded sequence with positions 0..N-1. */
    public float[][] encode(int[] inputIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        return encode(inputIds, null, sequentialPositions(inputIds.length));
    }

    /** Runs the model-card mean-pooling stage over the learned token projections. */
    public float[] encodePooled(int[] inputIds, boolean[] validTokens, int[] positionIds) {
        float[][] tokenStates = encode(inputIds, validTokens, positionIds);
        boolean[] poolingMask = validTokens == null ? allValid(inputIds.length) : validTokens;
        return EmbeddingGemma2MeanPooler.meanPool(tokenStates, poolingMask);
    }

    /** Convenience pooled path for an unpadded sequence. */
    public float[] encodePooled(int[] inputIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        return encodePooled(inputIds, null, sequentialPositions(inputIds.length));
    }

    public int hiddenSize() { return config.hiddenSize(); }
    public int embeddingDimension() { return config.embeddingDim(); }
    public int layerCount() { return layers.size(); }

    private float[][] embed(int[] inputIds) {
        float[][] result = new float[inputIds.length][config.hiddenSize()];
        float scale = (float) Math.sqrt(config.hiddenSize());
        float[][] table = weights.tokenEmbeddings();
        for (int token = 0; token < inputIds.length; token++) {
            int id = inputIds[token];
            if (id < 0 || id >= config.vocabSize()) {
                throw new IllegalArgumentException(
                        "token ID out of vocabulary range at index " + token + ": " + id);
            }
            for (int dimension = 0; dimension < config.hiddenSize(); dimension++) {
                result[token][dimension] = table[id][dimension] * scale;
            }
        }
        return result;
    }

    private void validateInputs(int[] inputIds, boolean[] validTokens, int[] positionIds) {
        if (inputIds == null || inputIds.length == 0) {
            throw new IllegalArgumentException("inputIds must be non-empty");
        }
        if (inputIds.length > config.maxInputTokens()) {
            throw new IllegalArgumentException("input sequence exceeds max_input_tokens: " + config.maxInputTokens());
        }
        if (validTokens != null && validTokens.length != inputIds.length) {
            throw new IllegalArgumentException("validTokens length must equal input sequence length");
        }
        if (positionIds == null || positionIds.length != inputIds.length) {
            throw new IllegalArgumentException("positionIds length must equal input sequence length");
        }
        for (int index = 0; index < positionIds.length; index++) {
            if (positionIds[index] < 0 || positionIds[index] >= config.maxPositionEmbeddings()) {
                throw new IllegalArgumentException("position ID out of range at index " + index);
            }
        }
    }

    private void validateWeights() {
        requireMatrix(weights.tokenEmbeddings(), config.vocabSize(), config.hiddenSize(), "tokenEmbeddings");
        requireMatrix(weights.pleProjection(),
                Math.multiplyExact(config.numHiddenLayers(), config.hiddenSizePerLayerInput()),
                config.hiddenSize(), "pleProjection");
        requireVector(weights.pleNormScale(), config.hiddenSizePerLayerInput(), "pleNormScale");
        requireVector(weights.finalNormScale(), config.hiddenSize(), "finalNormScale");
        requireMatrix(weights.embeddingProjection(), config.embeddingDim(), config.hiddenSize(),
                "embeddingProjection");
        if (weights.layers() == null || weights.layers().size() != config.numHiddenLayers()) {
            throw new IllegalArgumentException(
                    "exactly one checkpoint layer weight bundle is required per configured layer");
        }
    }

    private static int[] sequentialPositions(int length) {
        int[] positions = new int[length];
        for (int index = 0; index < length; index++) positions[index] = index;
        return positions;
    }

    private static boolean[] allValid(int length) {
        boolean[] mask = new boolean[length];
        java.util.Arrays.fill(mask, true);
        return mask;
    }

    private static void requireMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
            }
            for (float value : row) {
                if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must contain finite values");
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
        for (float value : vector) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must contain finite values");
        }
    }

    private static float[][] copy(float[][] input) {
        float[][] result = new float[input.length][];
        for (int index = 0; index < input.length; index++) result[index] = input[index].clone();
        return result;
    }

    /** Complete checkpoint tensor bundle; matrices use output-major [row][column] layout. */
    public record Weights(
            float[][] tokenEmbeddings,
            float[][] pleProjection,
            float[] pleNormScale,
            List<LayerWeights> layers,
            float[] finalNormScale,
            float[][] embeddingProjection) {
        public Weights {
            layers = layers == null ? null : List.copyOf(layers);
        }
    }

    /** Checkpoint tensors for one configured transformer layer. */
    public record LayerWeights(
            EmbeddingGemma2AttentionBlock.Weights attention,
            EmbeddingGemma2Mlp.Weights mlp,
            EmbeddingGemma2EncoderLayer.Weights encoder) {
    }
}
