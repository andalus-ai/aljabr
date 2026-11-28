package tech.kayys.aljabr.models.qwen3asr.encoder;

import java.util.Objects;

/**
 * CPU reference implementation of one Qwen3-ASR acoustic transformer layer.
 *
 * <p>The layer consumes packed token states and cumulative sequence boundaries.
 * Self-attention is bidirectional inside each boundary-delimited sequence and
 * never crosses an inference-window boundary. All parameters are supplied by
 * the checkpoint loader; this class does not initialize or own weight storage.</p>
 *
 * <p>This deliberately small reference path is for correctness and numerical
 * parity. Production execution should map the same operations to Aljabr tensor
 * and backend kernels rather than use these scalar loops.</p>
 */
public final class Qwen3AsrAudioEncoderLayer {
    private final int hiddenSize;
    private final int attentionHeads;
    private final int headSize;
    private final int feedForwardSize;
    private final Weights weights;

    public Qwen3AsrAudioEncoderLayer(int hiddenSize, int attentionHeads,
            int feedForwardSize, Weights weights) {
        if (hiddenSize <= 0 || attentionHeads <= 0 || hiddenSize % attentionHeads != 0
                || feedForwardSize <= 0) {
            throw new IllegalArgumentException("invalid Qwen3-ASR encoder layer dimensions");
        }
        this.hiddenSize = hiddenSize;
        this.attentionHeads = attentionHeads;
        this.headSize = hiddenSize / attentionHeads;
        this.feedForwardSize = feedForwardSize;
        this.weights = Objects.requireNonNull(weights, "weights");
        weights.validate(hiddenSize, feedForwardSize);
    }

    /** Creates the checkpoint-shaped layer for a model configuration. */
    public static Qwen3AsrAudioEncoderLayer from(
            tech.kayys.aljabr.models.qwen3asr.Qwen3AsrConfig.AudioEncoderConfig config,
            Weights weights) {
        Objects.requireNonNull(config, "config");
        return new Qwen3AsrAudioEncoderLayer(
                config.modelDimension(), config.attentionHeads(),
                config.feedForwardDimension(), weights);
    }

    /**
     * Computes one pre-norm encoder layer.
     *
     * @param hiddenStates packed input [totalTokens][hiddenSize]
     * @param cumulativeSequenceLengths monotonic boundaries beginning at zero
     *        and ending at totalTokens; each interval is an independent,
     *        bidirectional attention window
     * @return newly allocated output [totalTokens][hiddenSize]
     */
    public float[][] compute(float[][] hiddenStates, int[] cumulativeSequenceLengths) {
        validateInput(hiddenStates, cumulativeSequenceLengths);
        int tokenCount = hiddenStates.length;

        float[][] normalized = layerNorm(hiddenStates,
                weights.attentionNormScale, weights.attentionNormBias);
        float[][] queries = linear(normalized, weights.queryWeight, weights.queryBias);
        float[][] keys = linear(normalized, weights.keyWeight, weights.keyBias);
        float[][] values = linear(normalized, weights.valueWeight, weights.valueBias);
        float[][] attended = attend(queries, keys, values, cumulativeSequenceLengths);
        float[][] attentionOutput = linear(attended,
                weights.attentionOutputWeight, weights.attentionOutputBias);
        float[][] afterAttention = add(hiddenStates, attentionOutput);

        normalized = layerNorm(afterAttention,
                weights.feedForwardNormScale, weights.feedForwardNormBias);
        float[][] feedForward = linear(normalized,
                weights.feedForwardInputWeight, weights.feedForwardInputBias);
        geluInPlace(feedForward);
        feedForward = linear(feedForward,
                weights.feedForwardOutputWeight, weights.feedForwardOutputBias);
        float[][] output = add(afterAttention, feedForward);

        if (output.length != tokenCount) {
            throw new IllegalStateException("encoder layer changed token count");
        }
        return output;
    }

    private float[][] attend(float[][] q, float[][] k, float[][] v, int[] boundaries) {
        int tokens = q.length;
        float[][] result = new float[tokens][hiddenSize];
        float scale = (float) (1.0 / Math.sqrt(headSize));
        float[] logits = new float[tokens];

        for (int segment = 0; segment < boundaries.length - 1; segment++) {
            int start = boundaries[segment];
            int end = boundaries[segment + 1];
            for (int queryIndex = start; queryIndex < end; queryIndex++) {
                for (int head = 0; head < attentionHeads; head++) {
                    int offset = head * headSize;
                    float maxLogit = Float.NEGATIVE_INFINITY;
                    for (int keyIndex = start; keyIndex < end; keyIndex++) {
                        float dot = 0.0f;
                        for (int d = 0; d < headSize; d++) {
                            dot += q[queryIndex][offset + d] * k[keyIndex][offset + d];
                        }
                        float logit = dot * scale;
                        logits[keyIndex] = logit;
                        maxLogit = Math.max(maxLogit, logit);
                    }

                    double denominator = 0.0;
                    for (int keyIndex = start; keyIndex < end; keyIndex++) {
                        double probability = Math.exp(logits[keyIndex] - maxLogit);
                        logits[keyIndex] = (float) probability;
                        denominator += probability;
                    }
                    if (!(denominator > 0.0) || !Double.isFinite(denominator)) {
                        throw new ArithmeticException("attention softmax normalization is not finite");
                    }

                    for (int keyIndex = start; keyIndex < end; keyIndex++) {
                        float probability = (float) (logits[keyIndex] / denominator);
                        for (int d = 0; d < headSize; d++) {
                            result[queryIndex][offset + d] += probability * v[keyIndex][offset + d];
                        }
                    }
                }
            }
        }
        return result;
    }

    private float[][] layerNorm(float[][] input, float[] scale, float[] bias) {
        float[][] output = new float[input.length][hiddenSize];
        for (int token = 0; token < input.length; token++) {
            double mean = 0.0;
            for (float value : input[token]) mean += value;
            mean /= hiddenSize;
            double variance = 0.0;
            for (float value : input[token]) {
                double centered = value - mean;
                variance += centered * centered;
            }
            variance /= hiddenSize;
            double inverseStd = 1.0 / Math.sqrt(variance + weights.layerNormEpsilon);
            for (int d = 0; d < hiddenSize; d++) {
                output[token][d] = (float) (((input[token][d] - mean) * inverseStd)
                        * scale[d] + bias[d]);
            }
        }
        return output;
    }

    private static float[][] linear(float[][] input, float[][] weight, float[] bias) {
        int outputSize = weight.length;
        int inputSize = weight[0].length;
        float[][] output = new float[input.length][outputSize];
        for (int token = 0; token < input.length; token++) {
            for (int out = 0; out < outputSize; out++) {
                float sum = bias == null ? 0.0f : bias[out];
                for (int in = 0; in < inputSize; in++) {
                    sum += input[token][in] * weight[out][in];
                }
                output[token][out] = sum;
            }
        }
        return output;
    }

    private static void geluInPlace(float[][] values) {
        // Exact GELU: x * Phi(x), evaluated with a stable erf approximation.
        for (float[] row : values) {
            for (int i = 0; i < row.length; i++) {
                double x = row[i] / Math.sqrt(2.0);
                row[i] = (float) (row[i] * 0.5 * (1.0 + erf(x)));
            }
        }
    }

    private static double erf(double x) {
        double sign = Math.copySign(1.0, x);
        double z = Math.abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * z);
        double polynomial = (((((1.061405429 * t - 1.453152027) * t)
                + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        return sign * (1.0 - polynomial * Math.exp(-z * z));
    }

    private static float[][] add(float[][] left, float[][] right) {
        float[][] output = new float[left.length][left[0].length];
        for (int token = 0; token < left.length; token++) {
            for (int d = 0; d < left[token].length; d++) {
                output[token][d] = left[token][d] + right[token][d];
            }
        }
        return output;
    }

    private void validateInput(float[][] hiddenStates, int[] boundaries) {
        if (hiddenStates == null || hiddenStates.length == 0) {
            throw new IllegalArgumentException("hiddenStates must contain at least one token");
        }
        for (int token = 0; token < hiddenStates.length; token++) {
            if (hiddenStates[token] == null || hiddenStates[token].length != hiddenSize) {
                throw new IllegalArgumentException("hiddenStates[" + token + "] must have width " + hiddenSize);
            }
            for (float value : hiddenStates[token]) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("hiddenStates must contain only finite values");
                }
            }
        }
        if (boundaries == null || boundaries.length < 2 || boundaries[0] != 0
                || boundaries[boundaries.length - 1] != hiddenStates.length) {
            throw new IllegalArgumentException("sequence boundaries must start at zero and end at token count");
        }
        for (int i = 1; i < boundaries.length; i++) {
            if (boundaries[i] <= boundaries[i - 1]) {
                throw new IllegalArgumentException("sequence boundaries must be strictly increasing");
            }
        }
    }

    /**
     * Checkpoint tensors use [out_features][in_features] linear weight layout.
     * Arrays are retained by reference to avoid duplicating large checkpoint
     * allocations; callers must treat them as immutable for the layer lifetime.
     */
    public record Weights(
            float[] attentionNormScale,
            float[] attentionNormBias,
            float[][] queryWeight, float[] queryBias,
            float[][] keyWeight, float[] keyBias,
            float[][] valueWeight, float[] valueBias,
            float[][] attentionOutputWeight, float[] attentionOutputBias,
            float[] feedForwardNormScale, float[] feedForwardNormBias,
            float[][] feedForwardInputWeight, float[] feedForwardInputBias,
            float[][] feedForwardOutputWeight, float[] feedForwardOutputBias,
            double layerNormEpsilon) {

        public Weights {
            Objects.requireNonNull(attentionNormScale, "attentionNormScale");
            Objects.requireNonNull(attentionNormBias, "attentionNormBias");
            Objects.requireNonNull(queryWeight, "queryWeight");
            Objects.requireNonNull(queryBias, "queryBias");
            Objects.requireNonNull(keyWeight, "keyWeight");
            Objects.requireNonNull(keyBias, "keyBias");
            Objects.requireNonNull(valueWeight, "valueWeight");
            Objects.requireNonNull(valueBias, "valueBias");
            Objects.requireNonNull(attentionOutputWeight, "attentionOutputWeight");
            Objects.requireNonNull(attentionOutputBias, "attentionOutputBias");
            Objects.requireNonNull(feedForwardNormScale, "feedForwardNormScale");
            Objects.requireNonNull(feedForwardNormBias, "feedForwardNormBias");
            Objects.requireNonNull(feedForwardInputWeight, "feedForwardInputWeight");
            Objects.requireNonNull(feedForwardInputBias, "feedForwardInputBias");
            Objects.requireNonNull(feedForwardOutputWeight, "feedForwardOutputWeight");
            Objects.requireNonNull(feedForwardOutputBias, "feedForwardOutputBias");
            if (!Double.isFinite(layerNormEpsilon) || layerNormEpsilon <= 0.0) {
                throw new IllegalArgumentException("layerNormEpsilon must be finite and positive");
            }
        }

        private void validate(int hidden, int feedForward) {
            vector(attentionNormScale, hidden, "attentionNormScale");
            vector(attentionNormBias, hidden, "attentionNormBias");
            matrix(queryWeight, hidden, hidden, "queryWeight"); vector(queryBias, hidden, "queryBias");
            matrix(keyWeight, hidden, hidden, "keyWeight"); vector(keyBias, hidden, "keyBias");
            matrix(valueWeight, hidden, hidden, "valueWeight"); vector(valueBias, hidden, "valueBias");
            matrix(attentionOutputWeight, hidden, hidden, "attentionOutputWeight");
            vector(attentionOutputBias, hidden, "attentionOutputBias");
            vector(feedForwardNormScale, hidden, "feedForwardNormScale");
            vector(feedForwardNormBias, hidden, "feedForwardNormBias");
            matrix(feedForwardInputWeight, feedForward, hidden, "feedForwardInputWeight");
            vector(feedForwardInputBias, feedForward, "feedForwardInputBias");
            matrix(feedForwardOutputWeight, hidden, feedForward, "feedForwardOutputWeight");
            vector(feedForwardOutputBias, hidden, "feedForwardOutputBias");
        }

        private static void vector(float[] values, int expected, String name) {
            if (values.length != expected) throw new IllegalArgumentException(name + " must have length " + expected);
            for (float value : values) if (!Float.isFinite(value))
                throw new IllegalArgumentException(name + " must contain finite values");
        }

        private static void matrix(float[][] values, int rows, int columns, String name) {
            if (values.length != rows) throw new IllegalArgumentException(name + " must have " + rows + " rows");
            for (float[] row : values) {
                if (row == null || row.length != columns)
                    throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
                for (float value : row) if (!Float.isFinite(value))
                    throw new IllegalArgumentException(name + " must contain finite values");
            }
        }
    }
}
