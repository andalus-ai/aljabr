package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2LinearProjection;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2RmsNorm;

/**
 * Float32 reference implementation of Gemma4 vision self-attention.
 *
 * <p>Uses bias-free checkpoint-backed Q/K/V/O projections, learned Q/K
 * per-head RMSNorm, unscaled V RMSNorm, axial 2D RoPE on Q/K, and bidirectional
 * grouped-query attention. The attention scale is 1.0, matching the Gemma4
 * vision reference; it is deliberately not the usual {@code 1/sqrt(headDim)}.
 * There is no causal mask and no autoregressive KV cache.</p>
 *
 * <p>This implementation covers the default unclipped-linear configuration.
 * A checkpoint enabling clipped linears must supply and apply its clipping
 * bounds before using this operation.</p>
 */
public final class EmbeddingGemma2VisionAttention {
    private static final float ATTENTION_SCALE = 1.0f;

    private final int hiddenSize;
    private final int queryHeads;
    private final int keyValueHeads;
    private final int headDimension;
    private final float epsilon;
    private final EmbeddingGemma2VisionRotaryEmbedding rotaryEmbedding;
    private final Weights weights;

    public EmbeddingGemma2VisionAttention(
            int hiddenSize,
            int queryHeads,
            int keyValueHeads,
            int headDimension,
            float epsilon,
            double ropeTheta,
            Weights weights) {
        if (hiddenSize <= 0 || queryHeads <= 0 || keyValueHeads <= 0) {
            throw new IllegalArgumentException("hidden size and attention head counts must be positive");
        }
        if (queryHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException("queryHeads must be divisible by keyValueHeads");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        this.hiddenSize = hiddenSize;
        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
        this.headDimension = headDimension;
        this.epsilon = epsilon;
        this.rotaryEmbedding = new EmbeddingGemma2VisionRotaryEmbedding(headDimension, ropeTheta);
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        validateWeights();
    }

    /**
     * Executes one image's attention sequence.
     *
     * @param hiddenStates input [sequence][hiddenSize], already input-layer-normalized
     * @param pixelPositionIds patch (x,y) coordinates [sequence][2]
     * @param validPatches optional true=valid patch mask; false keys are excluded
     * @return projected output [sequence][hiddenSize]
     */
    public float[][] compute(float[][] hiddenStates, int[][] pixelPositionIds, boolean[] validPatches) {
        validateHiddenStates(hiddenStates);
        int sequence = hiddenStates.length;
        validatePositions(pixelPositionIds, sequence);
        if (validPatches != null && validPatches.length != sequence) {
            throw new IllegalArgumentException("validPatches length must equal sequence length");
        }
        boolean[] effectiveValidKeys = new boolean[sequence];
        int validCount = 0;
        for (int i = 0; i < sequence; i++) {
            boolean paddingPosition = pixelPositionIds[i][0] == -1;
            if (validPatches != null && validPatches[i] && paddingPosition) {
                throw new IllegalArgumentException("a padding position cannot be marked as a valid key");
            }
            effectiveValidKeys[i] = validPatches == null ? !paddingPosition : validPatches[i];
            if (effectiveValidKeys[i]) validCount++;
        }
        if (validCount == 0) {
            throw new IllegalArgumentException("vision attention requires at least one valid key patch");
        }

        float[][] queryFlat = EmbeddingGemma2LinearProjection.apply(hiddenStates, weights.query(), null);
        float[][] keyFlat = EmbeddingGemma2LinearProjection.apply(hiddenStates, weights.key(), null);
        float[][] valueFlat = EmbeddingGemma2LinearProjection.apply(hiddenStates, weights.value(), null);

        float[][][] query = reshape(queryFlat, queryHeads, headDimension);
        float[][][] key = reshape(keyFlat, keyValueHeads, headDimension);
        float[][][] value = reshape(valueFlat, keyValueHeads, headDimension);

        query = EmbeddingGemma2RmsNorm.apply(query, weights.queryNormScale(), epsilon);
        key = EmbeddingGemma2RmsNorm.apply(key, weights.keyNormScale(), epsilon);
        value = EmbeddingGemma2RmsNorm.apply(value, null, epsilon);

        query = rotaryEmbedding.apply(query, pixelPositionIds);
        key = rotaryEmbedding.apply(key, pixelPositionIds);

        float[][][] attended = attend(query, key, value, effectiveValidKeys);
        return EmbeddingGemma2LinearProjection.apply(flatten(attended), weights.output(), null);
    }

    private float[][][] attend(float[][][] query, float[][][] key, float[][][] value, boolean[] validPatches) {
        int sequence = query.length;
        int groups = queryHeads / keyValueHeads;
        float[][][] output = new float[sequence][queryHeads][headDimension];

        for (int head = 0; head < queryHeads; head++) {
            int kvHead = head / groups;
            for (int queryIndex = 0; queryIndex < sequence; queryIndex++) {
                double maxScore = Double.NEGATIVE_INFINITY;
                double[] scores = new double[sequence];
                for (int keyIndex = 0; keyIndex < sequence; keyIndex++) {
                    if (validPatches != null && !validPatches[keyIndex]) {
                        scores[keyIndex] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double dot = 0d;
                    for (int d = 0; d < headDimension; d++) {
                        dot += (double) query[queryIndex][head][d] * key[keyIndex][kvHead][d];
                    }
                    double score = dot * ATTENTION_SCALE;
                    scores[keyIndex] = score;
                    if (score > maxScore) maxScore = score;
                }

                double denominator = 0d;
                for (double score : scores) {
                    if (score != Double.NEGATIVE_INFINITY) denominator += Math.exp(score - maxScore);
                }
                for (int keyIndex = 0; keyIndex < sequence; keyIndex++) {
                    if (scores[keyIndex] == Double.NEGATIVE_INFINITY) continue;
                    float probability = (float) (Math.exp(scores[keyIndex] - maxScore) / denominator);
                    for (int d = 0; d < headDimension; d++) {
                        output[queryIndex][head][d] += probability * value[keyIndex][kvHead][d];
                    }
                }
            }
        }
        return output;
    }

    private void validateHiddenStates(float[][] states) {
        if (states == null || states.length == 0) {
            throw new IllegalArgumentException("hiddenStates must be non-empty");
        }
        for (float[] row : states) {
            if (row == null || row.length != hiddenSize) {
                throw new IllegalArgumentException("hiddenStates must have shape [sequence][" + hiddenSize + "]");
            }
            for (float value : row) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("hiddenStates must contain only finite values");
                }
            }
        }
    }

    private static void validatePositions(int[][] positions, int sequence) {
        if (positions == null || positions.length != sequence) {
            throw new IllegalArgumentException("pixelPositionIds length must equal sequence length");
        }
        boolean hasValidPatch = false;
        for (int i = 0; i < positions.length; i++) {
            if (positions[i] == null || positions[i].length != 2) {
                throw new IllegalArgumentException("pixel position " + i + " must contain (x,y)");
            }
            int x = positions[i][0], y = positions[i][1];
            if (!((x == -1 && y == -1) || (x >= 0 && y >= 0))) {
                throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
            }
            if (x >= 0) hasValidPatch = true;
        }
        if (!hasValidPatch) {
            throw new IllegalArgumentException("vision attention requires at least one valid patch position");
        }
    }

    private void validateWeights() {
        requireMatrix(weights.query(), queryHeads * headDimension, hiddenSize, "query");
        requireMatrix(weights.key(), keyValueHeads * headDimension, hiddenSize, "key");
        requireMatrix(weights.value(), keyValueHeads * headDimension, hiddenSize, "value");
        requireMatrix(weights.output(), hiddenSize, queryHeads * headDimension, "output");
        requireVector(weights.queryNormScale(), headDimension, "queryNormScale");
        requireVector(weights.keyNormScale(), headDimension, "keyNormScale");
    }

    private static void requireMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " weights must have shape [" + rows + "][" + columns + "]");
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " weights must have shape [" + rows + "][" + columns + "]");
            }
            for (float value : row) {
                if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " weights must be finite");
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
        for (float value : vector) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must be finite");
        }
    }

    private static float[][][] reshape(float[][] flat, int heads, int dimension) {
        float[][][] result = new float[flat.length][heads][dimension];
        for (int token = 0; token < flat.length; token++) {
            for (int head = 0; head < heads; head++) {
                System.arraycopy(flat[token], head * dimension, result[token][head], 0, dimension);
            }
        }
        return result;
    }

    private static float[][] flatten(float[][][] tensor) {
        int headDimension = tensor[0][0].length;
        int width = tensor[0].length * headDimension;
        float[][] result = new float[tensor.length][width];
        for (int token = 0; token < tensor.length; token++) {
            for (int head = 0; head < tensor[token].length; head++) {
                System.arraycopy(tensor[token][head], 0, result[token],
                        head * headDimension, headDimension);
            }
        }
        return result;
    }

    /** Bias-free checkpoint matrices in output-major [output][input] layout. */
    public record Weights(
            float[][] query,
            float[][] key,
            float[][] value,
            float[][] output,
            float[] queryNormScale,
            float[] keyNormScale) {
    }
}
