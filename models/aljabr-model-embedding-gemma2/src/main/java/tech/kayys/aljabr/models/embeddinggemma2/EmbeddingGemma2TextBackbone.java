package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2EncoderLayer;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2SequenceRmsNorm;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.output.EmbeddingGemma2MeanPooler;
import tech.kayys.aljabr.models.embeddinggemma2.output.EmbeddingGemma2Projection;
import tech.kayys.aljabr.models.embeddinggemma2.ple.EmbeddingGemma2PerLayerEmbeddingProjector;

/**
 * Composes the complete text-only EmbeddingGemma 2 backbone from checkpoint tensors.
 *
 * <p>Token embeddings are scaled by sqrt(hiddenSize), projection-only PLE is
 * generated from those scaled embeddings, every configured encoder layer is
 * executed in order, then final RMSNorm and the learned per-token embedding
 * projection are applied. Mean pooling and MRL normalization remain explicit
 * output-pipeline operations.</p>
 *
 * <p>This is a float32 reference path. It does not load checkpoints, tokenize
 * text, or implement the vision/audio towers.</p>
 */
public final class EmbeddingGemma2TextBackbone {
    private final EmbeddingGemma2Config config;
    private final List<EmbeddingGemma2EncoderLayer> layers;
    private final Weights weights;

    public EmbeddingGemma2TextBackbone(EmbeddingGemma2Config config,
            List<EmbeddingGemma2EncoderLayer> layers, Weights weights) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.layers = List.copyOf(Objects.requireNonNull(layers, "layers must not be null"));
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        if (this.layers.size() != config.numHiddenLayers()) {
            throw new IllegalArgumentException("encoder layer count must equal configured num_hidden_layers");
        }
        validateWeights();
    }

    /**
     * Runs token IDs through the complete text backbone.
     *
     * @param inputIds token IDs [sequence]
     * @param validKeys optional true=valid key mask; null means all valid
     * @param positionIds explicit position IDs [sequence]
     * @return projected per-token embeddings [sequence][embeddingDim]
     */
    public float[][] encode(int[] inputIds, boolean[] validKeys, int[] positionIds) {
        float[][] embeddings = embed(inputIds);
        int sequence = inputIds.length;
        if (positionIds == null || positionIds.length != sequence) {
            throw new IllegalArgumentException("positionIds length must equal input sequence length");
        }
        if (validKeys != null && validKeys.length != sequence) {
            throw new IllegalArgumentException("validKeys length must equal input sequence length");
        }

        float[][][] perLayerInputs = EmbeddingGemma2PerLayerEmbeddingProjector.project(
                embeddings, weights.pleProjection(), weights.pleNormScale(),
                config.numHiddenLayers(), config.hiddenSizePerLayerInput(), (float) config.rmsNormEps());

        float[][] hidden = copy(embeddings);
        for (int layer = 0; layer < layers.size(); layer++) {
            float[][] layerPle = new float[sequence][config.hiddenSizePerLayerInput()];
            for (int token = 0; token < sequence; token++) {
                System.arraycopy(perLayerInputs[token][layer], 0, layerPle[token], 0,
                        config.hiddenSizePerLayerInput());
            }
            hidden = layers.get(layer).compute(hidden, layerPle, positionIds, validKeys);
        }

        hidden = EmbeddingGemma2SequenceRmsNorm.apply(
                hidden, weights.finalNormScale(), (float) config.rmsNormEps());
        return projectTokens(hidden);
    }

    /** Convenience path for a single unpadded sequence with positions 0..N-1. */
    public float[][] encode(int[] inputIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        int[] positions = new int[inputIds.length];
        for (int i = 0; i < positions.length; i++) positions[i] = i;
        return encode(inputIds, null, positions);
    }

    /** Applies the model-card mean-pooling stage to projected token embeddings. */
    public float[] encodePooled(int[] inputIds, boolean[] validTokens, int[] positionIds) {
        return EmbeddingGemma2MeanPooler.meanPool(encode(inputIds, validTokens, positionIds), validTokens);
    }

    private float[][] embed(int[] inputIds) {
        if (inputIds == null || inputIds.length == 0) {
            throw new IllegalArgumentException("inputIds must be non-empty");
        }
        float[][] result = new float[inputIds.length][config.hiddenSize()];
        float scale = (float) Math.sqrt(config.hiddenSize());
        for (int token = 0; token < inputIds.length; token++) {
            int id = inputIds[token];
            if (id < 0 || id >= config.vocabSize()) {
                throw new IllegalArgumentException("token ID out of vocabulary range at index " + token + ": " + id);
            }
            for (int dimension = 0; dimension < config.hiddenSize(); dimension++) {
                result[token][dimension] = weights.tokenEmbeddings()[id][dimension] * scale;
            }
        }
        return result;
    }

    private float[][] projectTokens(float[][] hidden) {
        float[][] result = new float[hidden.length][config.embeddingDim()];
        for (int token = 0; token < hidden.length; token++) {
            result[token] = project(hidden[token], weights.embeddingProjection());
        }
        return result;
    }

    private float[] project(float[] input, float[][] projection) {
        float[] result = new float[config.embeddingDim()];
        for (int output = 0; output < result.length; output++) {
            double sum = 0d;
            for (int dimension = 0; dimension < config.hiddenSize(); dimension++) {
                sum += (double) projection[output][dimension] * input[dimension];
            }
            result[output] = (float) sum;
        }
        return result;
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
    }

    private static void requireMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
            }
            for (float value : row) if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain finite values");
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
        for (float value : vector) if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(name + " must contain finite values");
        }
    }

    private static float[][] copy(float[][] input) {
        float[][] result = new float[input.length][];
        for (int i = 0; i < input.length; i++) result[i] = input[i].clone();
        return result;
    }

    /** All arrays are checkpoint tensors; matrices use [row][column] layout. */
    public record Weights(float[][] tokenEmbeddings, float[][] pleProjection,
            float[] pleNormScale, float[] finalNormScale, float[][] embeddingProjection) {
    }
}
