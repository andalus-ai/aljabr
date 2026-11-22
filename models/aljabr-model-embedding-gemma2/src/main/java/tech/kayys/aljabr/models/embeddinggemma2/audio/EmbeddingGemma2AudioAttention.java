package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Checkpoint-backed Java reference for Gemma 4 chunked local audio attention.
 *
 * <p>Queries are split into non-overlapping chunks. Each chunk reads an
 * overlapping key/value context, applies the Transformer-XL relative shift,
 * per-dimension query scaling, capped logits, and a local validity mask.
 * Linear matrices use [out_features][in_features] layout. No parameters are
 * initialized or loaded by this class.</p>
 *
 * <p>This implementation is intended for numerical reference and parity work;
 * it does not replace Aljabr Tensor operations or accelerated backend kernels.</p>
 */
public final class EmbeddingGemma2AudioAttention {
    private final int hiddenSize;
    private final int numHeads;
    private final int headSize;
    private final int chunkSize;
    private final int maxPastHorizon;
    private final int maxFutureHorizon;
    private final int contextSize;
    private final float queryScale;
    private final float keyScale;
    private final float logitCap;
    private final float invalidLogit;
    private final boolean useClippedLinears;
    private final Weights weights;
    private final float[][] relativeKeys;

    public EmbeddingGemma2AudioAttention(EmbeddingGemma2AudioConfig config, Weights weights) {
        Objects.requireNonNull(config, "config must not be null");
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        this.hiddenSize = config.hiddenSize();
        this.numHeads = config.numAttentionHeads();
        this.headSize = hiddenSize / numHeads;
        this.chunkSize = config.attentionChunkSize();
        this.maxPastHorizon = config.attentionContextLeft() - 1;
        this.maxFutureHorizon = config.attentionContextRight();
        this.contextSize = Math.addExact(Math.addExact(chunkSize, maxPastHorizon), maxFutureHorizon);
        this.queryScale = (float) ((1.0 / Math.sqrt(headSize)) / Math.log(2.0));
        this.keyScale = (float) (Math.log1p(Math.E) / Math.log(2.0));
        this.logitCap = (float) config.attentionLogitCap();
        this.invalidLogit = (float) config.attentionInvalidLogitsValue();
        this.useClippedLinears = config.useClippedLinears();

        if (contextSize <= 0 || maxPastHorizon <= 0) {
            throw new IllegalArgumentException("audio attention requires a positive context window");
        }
        if (maxFutureHorizon != 0) {
            throw new IllegalArgumentException(
                    "non-zero right context is not yet supported by the Gemma 4 relative-shift reference path");
        }
        if ((contextSize & 1) != 0) {
            throw new IllegalArgumentException("audio attention context size must be even");
        }
        validateWeights();
        float[][] relativePositions = new EmbeddingGemma2AudioRelativePositionEncoding(config).compute();
        this.relativeKeys = linear(relativePositions, weights.relativeKeyProjection());
    }

    /**
     * Computes attention for hidden states [batch][sequence][hidden].
     *
     * @param hiddenStates input sequence
     * @param validMask optional true=valid mask [batch][sequence]
     * @return projected states and attention probabilities [batch][heads][sequence][context]
     */
    public Output compute(float[][][] hiddenStates, boolean[][] validMask) {
        validateInput(hiddenStates, validMask);
        int batch = hiddenStates.length;
        int sequence = hiddenStates[0].length;

        float[][][] queries = linear(hiddenStates, weights.query(), weights.query().clipRange());
        float[][][] keys = linear(hiddenStates, weights.key(), weights.key().clipRange());
        float[][][] values = linear(hiddenStates, weights.value(), weights.value().clipRange());

        scaleQueriesAndKeys(queries, keys);
        float[][][][] probabilities = new float[batch][numHeads][sequence][contextSize];
        float[][][] attended = new float[batch][sequence][hiddenSize];

        for (int b = 0; b < batch; b++) {
            for (int blockStart = 0; blockStart < sequence; blockStart += chunkSize) {
                for (int queryOffset = 0; queryOffset < chunkSize; queryOffset++) {
                    int queryIndex = blockStart + queryOffset;
                    if (queryIndex >= sequence || !isValid(validMask, b, queryIndex)) {
                        continue;
                    }
                    for (int head = 0; head < numHeads; head++) {
                        float[] logits = new float[contextSize];
                        boolean[] allowed = new boolean[contextSize];
                        int validCount = 0;
                        float max = Float.NEGATIVE_INFINITY;

                        for (int contextIndex = 0; contextIndex < contextSize; contextIndex++) {
                            int keyIndex = blockStart - maxPastHorizon + contextIndex;
                            int distance = queryIndex - keyIndex;
                            boolean valid = keyIndex >= 0 && keyIndex < sequence
                                    && isValid(validMask, b, keyIndex)
                                    && distance >= -maxFutureHorizon
                                    && distance < maxPastHorizon;
                            if (!valid) {
                                logits[contextIndex] = invalidLogit;
                                continue;
                            }
                            int relativeIndex = contextIndex - queryOffset;
                            if (relativeIndex < 0 || relativeIndex >= relativeKeys.length) {
                                throw new IllegalStateException("relative position index outside encoded context: "
                                        + relativeIndex);
                            }
                            float score = dotHead(queries[b][queryIndex], keys[b][keyIndex], head)
                                    + dotHead(queries[b][queryIndex], relativeKeys[relativeIndex], head);
                            score = (float) (logitCap * Math.tanh(score / logitCap));
                            logits[contextIndex] = score;
                            allowed[contextIndex] = true;
                            validCount++;
                            if (score > max) max = score;
                        }

                        if (validCount == 0) {
                            continue;
                        }
                        double denominator = 0.0;
                        for (int contextIndex = 0; contextIndex < contextSize; contextIndex++) {
                            if (allowed[contextIndex]) {
                                double exp = Math.exp(logits[contextIndex] - max);
                                probabilities[b][head][queryIndex][contextIndex] = (float) exp;
                                denominator += exp;
                            }
                        }
                        if (denominator == 0.0 || !Double.isFinite(denominator)) {
                            throw new ArithmeticException("audio attention softmax normalization failed");
                        }
                        for (int contextIndex = 0; contextIndex < contextSize; contextIndex++) {
                            if (!allowed[contextIndex]) continue;
                            float probability = (float) (probabilities[b][head][queryIndex][contextIndex] / denominator);
                            probabilities[b][head][queryIndex][contextIndex] = probability;
                            int keyIndex = blockStart - maxPastHorizon + contextIndex;
                            int headOffset = head * headSize;
                            for (int d = 0; d < headSize; d++) {
                                attended[b][queryIndex][headOffset + d] +=
                                        probability * values[b][keyIndex][headOffset + d];
                            }
                        }
                    }
                }
            }
        }

        float[][][] projected = linear(attended, weights.output(), weights.output().clipRange());
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < sequence; t++) {
                if (!isValid(validMask, b, t)) {
                    java.util.Arrays.fill(projected[b][t], 0f);
                }
            }
        }
        return new Output(projected, probabilities);
    }

    private void scaleQueriesAndKeys(float[][][] queries, float[][][] keys) {
        float[] scales = weights.perDimensionScale();
        for (int b = 0; b < queries.length; b++) {
            for (int t = 0; t < queries[b].length; t++) {
                for (int head = 0; head < numHeads; head++) {
                    int offset = head * headSize;
                    for (int d = 0; d < headSize; d++) {
                        int index = offset + d;
                        queries[b][t][index] *= queryScale * softplus(scales[d]);
                        keys[b][t][index] *= keyScale;
                    }
                }
            }
        }
    }

    private float[][][] linear(float[][][] input, LinearWeights linear, ClipRange range) {
        float[][] kernel = linear.kernel();
        float[][][] output = new float[input.length][input[0].length][kernel.length];
        for (int b = 0; b < input.length; b++) {
            for (int t = 0; t < input[b].length; t++) {
                for (int o = 0; o < kernel.length; o++) {
                    float sum = 0f;
                    for (int i = 0; i < input[b][t].length; i++) {
                        float value = input[b][t][i];
                        if (useClippedLinears && range != null) {
                            value = clamp(value, range.inputMin(), range.inputMax());
                        }
                        sum += value * kernel[o][i];
                    }
                    output[b][t][o] = useClippedLinears && range != null
                            ? clamp(sum, range.outputMin(), range.outputMax()) : sum;
                }
            }
        }
        return output;
    }

    private float[][] linear(float[][] input, float[][] kernel) {
        float[][] output = new float[input.length][kernel.length];
        for (int t = 0; t < input.length; t++) {
            for (int o = 0; o < kernel.length; o++) {
                float sum = 0f;
                for (int i = 0; i < input[t].length; i++) sum += input[t][i] * kernel[o][i];
                output[t][o] = sum;
            }
        }
        return output;
    }

    private float dotHead(float[] left, float[] right, int head) {
        int offset = head * headSize;
        float sum = 0f;
        for (int d = 0; d < headSize; d++) sum += left[offset + d] * right[offset + d];
        return sum;
    }

    private void validateWeights() {
        validateLinear(weights.query(), hiddenSize, "query");
        validateLinear(weights.key(), hiddenSize, "key");
        validateLinear(weights.value(), hiddenSize, "value");
        validateLinear(weights.output(), hiddenSize, "output");
        validateMatrix(weights.relativeKeyProjection(), hiddenSize, hiddenSize, "relativeKeyProjection");
        if (weights.perDimensionScale().length != headSize) {
            throw new IllegalArgumentException("perDimensionScale must have head dimension " + headSize);
        }
        validateFinite(weights.perDimensionScale(), "perDimensionScale");
    }

    private void validateLinear(LinearWeights linear, int inputWidth, String name) {
        Objects.requireNonNull(linear, name + " weights must not be null");
        validateMatrix(linear.kernel(), hiddenSize, inputWidth, name + ".kernel");
        ClipRange range = linear.clipRange();
        if (Double.isNaN(range.inputMin()) || Double.isNaN(range.inputMax())
                || Double.isNaN(range.outputMin()) || Double.isNaN(range.outputMax())
                || range.inputMin() > range.inputMax() || range.outputMin() > range.outputMax()) {
            throw new IllegalArgumentException(name + " clipping bounds must be ordered and not NaN");
        }
    }

    private void validateInput(float[][][] input, boolean[][] mask) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("hiddenStates must contain at least one batch item");
        }
        int sequence = -1;
        for (int b = 0; b < input.length; b++) {
            if (input[b] == null || input[b].length == 0) {
                throw new IllegalArgumentException("each audio item must contain at least one frame");
            }
            if (sequence < 0) sequence = input[b].length;
            if (input[b].length != sequence) throw new IllegalArgumentException("hiddenStates batch must be rectangular");
            if (mask != null && (mask.length != input.length || mask[b] == null || mask[b].length != sequence)) {
                throw new IllegalArgumentException("validMask must have shape [batch][sequence]");
            }
            for (float[] frame : input[b]) {
                if (frame == null || frame.length != hiddenSize) {
                    throw new IllegalArgumentException("each hidden state must have width " + hiddenSize);
                }
                validateFinite(frame, "hiddenStates");
            }
        }
        if (mask != null && mask.length != input.length) {
            throw new IllegalArgumentException("validMask batch dimension must match hiddenStates");
        }
    }

    private static boolean isValid(boolean[][] mask, int batch, int time) {
        return mask == null || mask[batch][time];
    }

    private static float softplus(float value) {
        return (float) (Math.max(value, 0f) + Math.log1p(Math.exp(-Math.abs(value))));
    }

    private static float clamp(float value, double min, double max) {
        return (float) Math.max(min, Math.min(max, value));
    }

    private static void validateMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix.length != rows) throw new IllegalArgumentException(name + " must have " + rows + " rows");
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " rows must have width " + columns);
            }
            validateFinite(row, name);
        }
    }

    private static void validateFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must contain finite values");
        }
    }

    /** Checkpoint-backed linear matrix with optional per-layer activation clipping bounds. */
    public record LinearWeights(float[][] kernel, ClipRange clipRange) {
        public LinearWeights {
            Objects.requireNonNull(kernel, "kernel must not be null");
            Objects.requireNonNull(clipRange, "clipRange must not be null");
            kernel = copy(kernel);
        }
        @Override public float[][] kernel() { return copy(kernel); }
    }

    public record ClipRange(double inputMin, double inputMax, double outputMin, double outputMax) {
        public static ClipRange unbounded() {
            return new ClipRange(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
    }

    /** All arrays are checkpoint tensors; matrices use [out][in] layout. */
    public record Weights(
            LinearWeights query,
            LinearWeights key,
            LinearWeights value,
            LinearWeights output,
            float[][] relativeKeyProjection,
            float[] perDimensionScale) {
        public Weights {
            Objects.requireNonNull(query, "query must not be null");
            Objects.requireNonNull(key, "key must not be null");
            Objects.requireNonNull(value, "value must not be null");
            Objects.requireNonNull(output, "output must not be null");
            relativeKeyProjection = copy(relativeKeyProjection);
            perDimensionScale = Objects.requireNonNull(perDimensionScale, "perDimensionScale must not be null").clone();
        }
        @Override public float[][] relativeKeyProjection() { return copy(relativeKeyProjection); }
        @Override public float[] perDimensionScale() { return perDimensionScale.clone(); }
    }

    public record Output(float[][][] hiddenStates, float[][][][] attentionWeights) {
        public Output {
            hiddenStates = copy(hiddenStates);
            attentionWeights = copy(attentionWeights);
        }
        @Override public float[][][] hiddenStates() { return copy(hiddenStates); }
        @Override public float[][][][] attentionWeights() { return copy(attentionWeights); }
    }

    private static float[][] copy(float[][] source) {
        Objects.requireNonNull(source, "matrix must not be null");
        float[][] result = new float[source.length][];
        for (int i = 0; i < source.length; i++) result[i] = source[i] == null ? null : source[i].clone();
        return result;
    }

    private static float[][][] copy(float[][][] source) {
        Objects.requireNonNull(source, "tensor must not be null");
        float[][][] result = new float[source.length][][];
        for (int i = 0; i < source.length; i++) {
            result[i] = new float[source[i].length][];
            for (int j = 0; j < source[i].length; j++) result[i][j] = source[i][j].clone();
        }
        return result;
    }

    private static float[][][][] copy(float[][][][] source) {
        Objects.requireNonNull(source, "tensor must not be null");
        float[][][][] result = new float[source.length][][][];
        for (int i = 0; i < source.length; i++) {
            result[i] = new float[source[i].length][][];
            for (int j = 0; j < source[i].length; j++) {
                result[i][j] = new float[source[i][j].length][];
                for (int k = 0; k < source[i][j].length; k++) {
                    result[i][j][k] = source[i][j][k].clone();
                }
            }
        }
        return result;
    }
}
