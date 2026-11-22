package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Checkpoint-backed Gemma 4 audio Conformer feed-forward residual block.
 *
 * <p>The block follows the published sequence:
 * clamp -> RMSNorm -> clipped linear (4x expansion) -> SiLU -> clipped
 * linear (projection) -> clamp -> RMSNorm -> residual scaling -> residual
 * addition. Computation is performed in Java float precision; checkpoint
 * tensors are supplied in logical [out][in] layout.</p>
 *
 * <p>This component owns no checkpoint loading and does not initialize
 * parameters. It is a deterministic reference implementation for parity
 * testing before backend-kernel integration.</p>
 */
public final class EmbeddingGemma2AudioFeedForward {
    private static final float GRADIENT_CLIPPING = 1.0e10f;
    private final int hiddenSize;
    private final int intermediateSize;
    private final float epsilon;
    private final float residualWeight;
    private final Weights weights;

    public EmbeddingGemma2AudioFeedForward(
            EmbeddingGemma2AudioConfig config, Weights weights) {
        Objects.requireNonNull(config, "config must not be null");
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        this.hiddenSize = config.hiddenSize();
        this.intermediateSize = Math.multiplyExact(hiddenSize, 4);
        this.epsilon = (float) config.rmsNormEps();
        this.residualWeight = (float) config.residualWeight();
        validateWeights();
    }

    /** Applies the block to hidden states shaped [batch][sequence][hidden]. */
    public float[][][] compute(float[][][] input) {
        validateInput(input);
        float[][][] residual = copy(input);
        float[][][] hidden = clamp(input, GRADIENT_CLIPPING);
        hidden = rmsNorm(hidden, weights.preNormScale(), epsilon);
        hidden = linear(hidden, weights.expand(), GRADIENT_CLIPPING);
        for (int b = 0; b < hidden.length; b++) {
            for (int t = 0; t < hidden[b].length; t++) {
                for (int i = 0; i < intermediateSize; i++) {
                    float value = hidden[b][t][i];
                    hidden[b][t][i] = (float) (value / (1.0 + Math.exp(-value)));
                }
            }
        }
        hidden = linear(hidden, weights.contract(), GRADIENT_CLIPPING);
        hidden = clamp(hidden, GRADIENT_CLIPPING);
        hidden = rmsNorm(hidden, weights.postNormScale(), epsilon);

        for (int b = 0; b < hidden.length; b++) {
            for (int t = 0; t < hidden[b].length; t++) {
                for (int h = 0; h < hiddenSize; h++) {
                    hidden[b][t][h] = residual[b][t][h] + residualWeight * hidden[b][t][h];
                }
            }
        }
        return hidden;
    }

    private float[][][] linear(float[][][] input, LinearWeights linear, float clipping) {
        float[][][] output = new float[input.length][input[0].length][linear.kernel().length];
        ClipRange range = linear.clipRange();
        for (int b = 0; b < input.length; b++) {
            for (int t = 0; t < input[b].length; t++) {
                for (int o = 0; o < linear.kernel().length; o++) {
                    float sum = 0f;
                    for (int i = 0; i < input[b][t].length; i++) {
                        float value = clamp(input[b][t][i], range.inputMin(), range.inputMax());
                        value = clamp(value, clipping);
                        sum += value * linear.kernel()[o][i];
                    }
                    output[b][t][o] = clamp(clamp(sum, range.outputMin(), range.outputMax()), clipping);
                }
            }
        }
        return output;
    }

    private static float[][][] rmsNorm(float[][][] input, float[] scale, float epsilon) {
        float[][][] output = new float[input.length][input[0].length][scale.length];
        for (int b = 0; b < input.length; b++) {
            for (int t = 0; t < input[b].length; t++) {
                double meanSquare = 0.0;
                for (float value : input[b][t]) meanSquare += (double) value * value;
                meanSquare /= scale.length;
                float inverseRms = (float) (1.0 / Math.sqrt(meanSquare + epsilon));
                for (int h = 0; h < scale.length; h++) {
                    output[b][t][h] = input[b][t][h] * inverseRms * scale[h];
                }
            }
        }
        return output;
    }

    private void validateWeights() {
        validateMatrix(weights.expand().kernel(), intermediateSize, hiddenSize, "expand");
        validateMatrix(weights.contract().kernel(), hiddenSize, intermediateSize, "contract");
        validateVector(weights.preNormScale(), hiddenSize, "preNormScale");
        validateVector(weights.postNormScale(), hiddenSize, "postNormScale");
        validateRange(weights.expand().clipRange(), "expand");
        validateRange(weights.contract().clipRange(), "contract");
    }

    private static void validateMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix.length != rows) throw new IllegalArgumentException(name + " output dimension must be " + rows);
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " input dimension must be " + columns);
            }
            for (float value : row) {
                if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " weights must be finite");
            }
        }
    }

    private static void validateVector(float[] vector, int size, String name) {
        if (vector.length != size) throw new IllegalArgumentException(name + " length must be " + size);
        for (float value : vector) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " values must be finite");
        }
    }

    private static void validateRange(ClipRange range, String name) {
        if (Double.isNaN(range.inputMin()) || Double.isNaN(range.inputMax())
                || Double.isNaN(range.outputMin()) || Double.isNaN(range.outputMax())
                || range.inputMin() > range.inputMax() || range.outputMin() > range.outputMax()) {
            throw new IllegalArgumentException(name + " clipping bounds must be ordered and not NaN");
        }
    }

    private void validateInput(float[][][] input) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("input must contain at least one batch item");
        }
        int sequence = -1;
        for (float[][] batch : input) {
            if (batch == null || batch.length == 0) {
                throw new IllegalArgumentException("each batch item must contain at least one token");
            }
            if (sequence < 0) sequence = batch.length;
            if (batch.length != sequence) throw new IllegalArgumentException("input batch must be rectangular");
            for (float[] token : batch) {
                if (token == null || token.length != hiddenSize) {
                    throw new IllegalArgumentException("each token must have hidden size " + hiddenSize);
                }
            }
        }
    }

    private static float clamp(float value, double min, double max) {
        return (float) Math.max(min, Math.min(max, value));
    }

    private static float clamp(float value, float limit) {
        return clamp(value, -limit, limit);
    }

    private static float[][][] clamp(float[][][] input, float limit) {
        float[][][] output = copy(input);
        for (int b = 0; b < output.length; b++) {
            for (int t = 0; t < output[b].length; t++) {
                for (int h = 0; h < output[b][t].length; h++) {
                    output[b][t][h] = clamp(output[b][t][h], limit);
                }
            }
        }
        return output;
    }

    private static float[][][] copy(float[][][] input) {
        float[][][] result = new float[input.length][][];
        for (int b = 0; b < input.length; b++) {
            result[b] = new float[input[b].length][];
            for (int t = 0; t < input[b].length; t++) result[b][t] = input[b][t].clone();
        }
        return result;
    }

    /** Checkpoint linear weight matrix, in [out_features][in_features] order. */
    public record LinearWeights(float[][] kernel, ClipRange clipRange) {
        public LinearWeights {
            Objects.requireNonNull(kernel, "kernel must not be null");
            Objects.requireNonNull(clipRange, "clipRange must not be null");
            float[][] copy = new float[kernel.length][];
            for (int i = 0; i < kernel.length; i++) copy[i] = kernel[i] == null ? null : kernel[i].clone();
            kernel = copy;
        }
        @Override public float[][] kernel() {
            float[][] copy = new float[kernel.length][];
            for (int i = 0; i < kernel.length; i++) copy[i] = kernel[i] == null ? null : kernel[i].clone();
            return copy;
        }
    }

    /** Per-layer checkpoint clipping bounds used by Gemma4ClippableLinear. */
    public record ClipRange(double inputMin, double inputMax, double outputMin, double outputMax) {
        public static ClipRange unbounded() {
            return new ClipRange(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
    }

    /** Parameters are supplied by the checkpoint mapper; no implicit initialization. */
    public record Weights(
            LinearWeights expand,
            LinearWeights contract,
            float[] preNormScale,
            float[] postNormScale) {
        public Weights {
            Objects.requireNonNull(expand, "expand must not be null");
            Objects.requireNonNull(contract, "contract must not be null");
            preNormScale = Objects.requireNonNull(preNormScale, "preNormScale must not be null").clone();
            postNormScale = Objects.requireNonNull(postNormScale, "postNormScale must not be null").clone();
        }
        @Override public float[] preNormScale() { return preNormScale.clone(); }
        @Override public float[] postNormScale() { return postNormScale.clone(); }
    }
}
