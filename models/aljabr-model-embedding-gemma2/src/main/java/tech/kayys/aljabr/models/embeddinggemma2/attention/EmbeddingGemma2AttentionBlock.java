package tech.kayys.aljabr.models.embeddinggemma2.attention;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfig;

/**
 * Float32 reference implementation of one EmbeddingGemma 2 attention sub-block.
 *
 * <p>Consumes hidden states after the encoder's input layer norm and applies
 * checkpoint-backed Q/K/V/O projections, per-head RMSNorm, Q/K RoPE, and
 * bidirectional grouped-query attention. The encoder residual and MLP stages
 * are intentionally outside this class.</p>
 */
public final class EmbeddingGemma2AttentionBlock {

    private final int hiddenSize;
    private final EmbeddingGemma2LayerConfig layer;
    private final Weights weights;
    private final double rmsNormEpsilon;

    public EmbeddingGemma2AttentionBlock(int hiddenSize, EmbeddingGemma2LayerConfig layer,
            Weights weights, double rmsNormEpsilon) {
        if (hiddenSize <= 0) throw new IllegalArgumentException("hiddenSize must be positive");
        this.layer = Objects.requireNonNull(layer, "layer config must not be null");
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        if (!Double.isFinite(rmsNormEpsilon) || rmsNormEpsilon <= 0.0d) {
            throw new IllegalArgumentException("rmsNormEpsilon must be finite and positive");
        }
        this.hiddenSize = hiddenSize;
        this.rmsNormEpsilon = rmsNormEpsilon;
        validateWeights();
    }

    /**
     * Executes attention for [sequence][hiddenSize] states.
     *
     * @param hiddenStates normalized input states
     * @param positionIds explicit positions used by RoPE and local attention
     * @param validKeys optional mask where true denotes a non-padding key
     */
    public float[][] compute(float[][] hiddenStates, int[] positionIds, boolean[] validKeys) {
        validateHidden(hiddenStates);
        int sequence = hiddenStates.length;
        if (positionIds == null || positionIds.length != sequence) {
            throw new IllegalArgumentException("positionIds length must equal sequence length");
        }

        float[][] qFlat = EmbeddingGemma2LinearProjection.apply(hiddenStates,
                weights.query(), weights.queryBias());
        float[][] kFlat = EmbeddingGemma2LinearProjection.apply(hiddenStates,
                weights.key(), weights.keyBias());
        float[][] vFlat = EmbeddingGemma2LinearProjection.apply(hiddenStates,
                weights.value(), weights.valueBias());

        float[][][] query = reshape(qFlat, layer.numAttentionHeads(), layer.headDim());
        float[][][] key = reshape(kFlat, layer.numKeyValueHeads(), layer.headDim());
        float[][][] value = reshape(vFlat, layer.numKeyValueHeads(), layer.headDim());

        query = EmbeddingGemma2RmsNorm.apply(query, weights.queryNormScale(), (float) rmsNormEpsilon);
        key = EmbeddingGemma2RmsNorm.apply(key, weights.keyNormScale(), (float) rmsNormEpsilon);
        value = EmbeddingGemma2RmsNorm.apply(value, null, (float) rmsNormEpsilon);

        query = EmbeddingGemma2RotaryEmbedding.apply(
                query, positionIds, layer.ropeTheta(), layer.partialRotaryFactor());
        key = EmbeddingGemma2RotaryEmbedding.apply(
                key, positionIds, layer.ropeTheta(), layer.partialRotaryFactor());

        Integer radius = layer.attentionType() == EmbeddingGemma2LayerConfig.AttentionType.SLIDING
                ? layer.slidingWindow() : null;
        float[][][] attended = EmbeddingGemma2BidirectionalAttention.compute(
                query, key, value, positionIds, validKeys, radius, 1.0f).output();

        float[][] flattened = flatten(attended);
        return EmbeddingGemma2LinearProjection.apply(flattened, weights.output(), weights.outputBias());
    }

    private void validateHidden(float[][] hiddenStates) {
        if (hiddenStates == null || hiddenStates.length == 0) {
            throw new IllegalArgumentException("hiddenStates must be non-empty");
        }
        for (float[] token : hiddenStates) {
            if (token == null || token.length != hiddenSize) {
                throw new IllegalArgumentException("hiddenStates must have shape [sequence][" + hiddenSize + "]");
            }
        }
    }

    private void validateWeights() {
        requireMatrix(weights.query(), layer.queryProjectionSize(), hiddenSize, "query");
        requireMatrix(weights.key(), layer.keyValueProjectionSize(), hiddenSize, "key");
        requireMatrix(weights.value(), layer.keyValueProjectionSize(), hiddenSize, "value");
        requireMatrix(weights.output(), hiddenSize, layer.queryProjectionSize(), "output");
        requireVector(weights.queryNormScale(), layer.headDim(), "queryNormScale");
        requireVector(weights.keyNormScale(), layer.headDim(), "keyNormScale");
        validateBias(weights.queryBias(), layer.queryProjectionSize(), "queryBias");
        validateBias(weights.keyBias(), layer.keyValueProjectionSize(), "keyBias");
        validateBias(weights.valueBias(), layer.keyValueProjectionSize(), "valueBias");
        validateBias(weights.outputBias(), hiddenSize, "outputBias");
    }

    private static void requireMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " weights must have shape [" + rows + "][" + columns + "]");
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " weights must have shape [" + rows + "][" + columns + "]");
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
    }

    private static void validateBias(float[] bias, int length, String name) {
        if (bias != null && bias.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
    }

    private static float[][][] reshape(float[][] flat, int heads, int headDim) {
        float[][][] result = new float[flat.length][heads][headDim];
        for (int token = 0; token < flat.length; token++) {
            for (int head = 0; head < heads; head++) {
                System.arraycopy(flat[token], head * headDim, result[token][head], 0, headDim);
            }
        }
        return result;
    }

    private static float[][] flatten(float[][][] tensor) {
        int width = tensor[0].length * tensor[0][0].length;
        float[][] result = new float[tensor.length][width];
        for (int token = 0; token < tensor.length; token++) {
            for (int head = 0; head < tensor[token].length; head++) {
                System.arraycopy(tensor[token][head], 0, result[token],
                        head * tensor[token][head].length, tensor[token][head].length);
            }
        }
        return result;
    }

    /** Checkpoint tensors; matrices are output-major and bias is optional. */
    public record Weights(
            float[][] query, float[] queryBias,
            float[][] key, float[] keyBias,
            float[][] value, float[] valueBias,
            float[][] output, float[] outputBias,
            float[] queryNormScale, float[] keyNormScale) {
    }
}
