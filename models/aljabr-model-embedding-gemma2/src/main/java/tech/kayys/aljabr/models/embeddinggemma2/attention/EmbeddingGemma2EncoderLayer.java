package tech.kayys.aljabr.models.embeddinggemma2.attention;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfig;
import tech.kayys.aljabr.models.embeddinggemma2.ple.EmbeddingGemma2PerLayerResidual;

/**
 * Complete text encoder layer reference for EmbeddingGemma 2.
 *
 * <p>Implements the checkpoint-defined sequence:
 * input RMSNorm -> bidirectional attention -> post-attention RMSNorm + residual;
 * pre-feedforward RMSNorm -> gated GELU MLP -> post-feedforward RMSNorm + residual;
 * PLE residual block -> fixed layer scalar (one).</p>
 *
 * <p>All trainable tensors are supplied by the checkpoint loader. The
 * architecture's layer_scalar is a non-trainable buffer initialized to one.</p>
 */
public final class EmbeddingGemma2EncoderLayer {

    private final int hiddenSize;
    private final double epsilon;
    private final EmbeddingGemma2AttentionBlock attention;
    private final EmbeddingGemma2Mlp mlp;
    private final Weights weights;

    public EmbeddingGemma2EncoderLayer(EmbeddingGemma2Config config,
            EmbeddingGemma2LayerConfig layerConfig,
            EmbeddingGemma2AttentionBlock.Weights attentionWeights,
            EmbeddingGemma2Mlp.Weights mlpWeights,
            Weights weights) {
        Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(layerConfig, "layerConfig must not be null");
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        this.hiddenSize = config.hiddenSize();
        this.epsilon = config.rmsNormEps();
        if (!EmbeddingGemma2Config.DEFAULT_HIDDEN_ACTIVATION.equals(config.hiddenActivation())) {
            throw new IllegalArgumentException("unsupported activation: " + config.hiddenActivation());
        }
        this.attention = new EmbeddingGemma2AttentionBlock(
                hiddenSize, layerConfig, attentionWeights, epsilon);
        this.mlp = new EmbeddingGemma2Mlp(hiddenSize, config.intermediateSize(), mlpWeights);
        validateWeights(config.hiddenSize(), config.hiddenSizePerLayerInput());
    }

    /**
     * @param hiddenStates input residual stream [sequence][hiddenSize]
     * @param perLayerInput this layer's projected PLE slice [sequence][pleSize]
     * @param positionIds explicit position IDs
     * @param validKeys optional true=valid key mask
     */
    public float[][] compute(float[][] hiddenStates, float[][] perLayerInput,
            int[] positionIds, boolean[] validKeys) {
        validateMatrix(hiddenStates, hiddenSize, "hiddenStates");
        validateMatrix(perLayerInput, weights.pleNormScale().length, "perLayerInput");
        if (perLayerInput.length != hiddenStates.length) {
            throw new IllegalArgumentException("perLayerInput sequence length must match hiddenStates");
        }

        float[][] residual = copy(hiddenStates);
        float[][] normalized = EmbeddingGemma2SequenceRmsNorm.apply(
                hiddenStates, weights.inputNormScale(), (float) epsilon);
        float[][] attended = attention.compute(normalized, positionIds, validKeys);
        attended = EmbeddingGemma2SequenceRmsNorm.apply(
                attended, weights.postAttentionNormScale(), (float) epsilon);
        float[][] afterAttention = add(residual, attended);

        residual = copy(afterAttention);
        normalized = EmbeddingGemma2SequenceRmsNorm.apply(
                afterAttention, weights.preFeedforwardNormScale(), (float) epsilon);
        float[][] feedforward = mlp.compute(normalized);
        feedforward = EmbeddingGemma2SequenceRmsNorm.apply(
                feedforward, weights.postFeedforwardNormScale(), (float) epsilon);
        float[][] afterFeedforward = add(residual, feedforward);

        float[][] afterPle = EmbeddingGemma2PerLayerResidual.apply(
                afterFeedforward, perLayerInput,
                weights.pleGate(), weights.pleProjection(), weights.pleNormScale(), (float) epsilon);

        // HF's layer_scalar is a non-trainable buffer initialized to one.
        return afterPle;
    }

    private void validateWeights(int hidden, int ple) {
        requireVector(weights.inputNormScale(), hidden, "inputNormScale");
        requireVector(weights.postAttentionNormScale(), hidden, "postAttentionNormScale");
        requireVector(weights.preFeedforwardNormScale(), hidden, "preFeedforwardNormScale");
        requireVector(weights.postFeedforwardNormScale(), hidden, "postFeedforwardNormScale");
        requireMatrix(weights.pleGate(), ple, hidden, "pleGate");
        requireMatrix(weights.pleProjection(), hidden, ple, "pleProjection");
        requireVector(weights.pleNormScale(), hidden, "pleNormScale");
    }

    private static void validateMatrix(float[][] matrix, int width, String name) {
        if (matrix == null || matrix.length == 0) throw new IllegalArgumentException(name + " must be non-empty");
        for (float[] row : matrix) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException(name + " must have feature width " + width);
            }
        }
    }

    private static void requireMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
    }

    private static float[][] add(float[][] left, float[][] right) {
        float[][] output = new float[left.length][left[0].length];
        for (int token = 0; token < left.length; token++) {
            for (int d = 0; d < left[token].length; d++) output[token][d] = left[token][d] + right[token][d];
        }
        return output;
    }

    private static float[][] copy(float[][] input) {
        float[][] output = new float[input.length][];
        for (int i = 0; i < input.length; i++) output[i] = input[i].clone();
        return output;
    }

    /** Trainable encoder tensors, all copied from the model checkpoint. */
    public record Weights(
            float[] inputNormScale,
            float[] postAttentionNormScale,
            float[] preFeedforwardNormScale,
            float[] postFeedforwardNormScale,
            float[][] pleGate,
            float[][] pleProjection,
            float[] pleNormScale) {
    }
}
