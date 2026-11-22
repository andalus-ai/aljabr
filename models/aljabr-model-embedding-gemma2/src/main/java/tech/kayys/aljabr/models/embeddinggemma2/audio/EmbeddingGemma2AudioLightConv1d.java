package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Checkpoint-backed reference implementation of Gemma 4's audio light
 * Conformer convolution block.
 *
 * <p>The operation order matches the published implementation:
 * pre-RMSNorm -> clipped expansion linear -> GLU -> causal depthwise Conv1D
 * -> gradient clamp -> convolution RMSNorm -> SiLU -> clipped output linear
 * -> residual addition. Convolution weights use [hidden][kernel] layout and
 * linear weights use [out][in] layout. No parameter initialization or
 * checkpoint loading is performed here.</p>
 *
 * <p>This is a Java float reference path for correctness/parity work. It is
 * not a replacement for Aljabr Tensor/backend kernels.</p>
 */
public final class EmbeddingGemma2AudioLightConv1d {
    private final int hiddenSize;
    private final int kernelSize;
    private final float epsilon;
    private final float gradientClipping;
    private final boolean useClippedLinears;
    private final Weights weights;

    public EmbeddingGemma2AudioLightConv1d(
            EmbeddingGemma2AudioConfig config, Weights weights) {
        Objects.requireNonNull(config, "config must not be null");
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        this.hiddenSize = config.hiddenSize();
        this.kernelSize = config.convolutionKernelSize();
        this.epsilon = (float) config.rmsNormEps();
        this.gradientClipping = (float) Math.min(config.gradientClipping(), Float.MAX_VALUE);
        this.useClippedLinears = config.useClippedLinears();
        validateWeights();
    }

    /** Applies the block to hidden states shaped [batch][sequence][hidden]. */
    public float[][][] compute(float[][][] input) {
        validateInput(input);
        float[][][] residual = copy(input);

        float[][][] hidden = rmsNorm(input, weights.preNormScale(), epsilon);
        hidden = linear(hidden, weights.linearStart());
        hidden = gatedLinearUnit(hidden);
        hidden = causalDepthwiseConvolution(hidden);
        hidden = clamp(hidden, gradientClipping);
        hidden = rmsNorm(hidden, weights.convolutionNormScale(), epsilon);
        hidden = silu(hidden);
        hidden = linear(hidden, weights.linearEnd());

        for (int b = 0; b < hidden.length; b++) {
            for (int t = 0; t < hidden[b].length; t++) {
                for (int h = 0; h < hiddenSize; h++) {
                    hidden[b][t][h] += residual[b][t][h];
                }
            }
        }
        return hidden;
    }

    private float[][][] linear(float[][][] input, LinearWeights linear) {
        float[][] kernel = linear.kernel();
        float[][][] output = new float[input.length][input[0].length][kernel.length];
        ClipRange range = linear.clipRange();
        for (int b = 0; b < input.length; b++) {
            for (int t = 0; t < input[b].length; t++) {
                for (int o = 0; o < kernel.length; o++) {
                    float sum = 0f;
                    for (int i = 0; i < input[b][t].length; i++) {
                        float value = input[b][t][i];
                        if (useClippedLinears) {
                            value = clamp(value, range.inputMin(), range.inputMax());
                        }
                        sum += value * kernel[o][i];
                    }
                    output[b][t][o] = useClippedLinears
                            ? clamp(sum, range.outputMin(), range.outputMax()) : sum;
                }
            }
        }
        return output;
    }

    private float[][][] causalDepthwiseConvolution(float[][][] input) {
        float[][][] output = new float[input.length][input[0].length][hiddenSize];
        int leftPadding = kernelSize - 1;
        float[][] kernels = weights.depthwiseKernel();
        for (int b = 0; b < input.length; b++) {
            for (int t = 0; t < input[b].length; t++) {
                for (int h = 0; h < hiddenSize; h++) {
                    float sum = 0f;
                    for (int k = 0; k < kernelSize; k++) {
                        int sourceTime = t + k - leftPadding;
                        if (sourceTime >= 0) {
                            sum += input[b][sourceTime][h] * kernels[h][k];
                        }
                    }
                    output[b][t][h] = sum;
                }
            }
        }
        return output;
    }

    private static float[][][] gatedLinearUnit(float[][][] input) {
        int hidden = input[0][0].length / 2;
        float[][][] output = new float[input.length][input[0].length][hidden];
        for (int b = 0; b < input.length; b++) {
            for (int t = 0; t < input[b].length; t++) {
                for (int h = 0; h < hidden; h++) {
                    float gate = input[b][t][h];
                    float linear = input[b][t][hidden + h];
                    output[b][t][h] = gate / (1.0f + (float) Math.exp(-linear));
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

    private static float[][][] silu(float[][][] input) {
        float[][][] output = copy(input);
        for (int b = 0; b < output.length; b++) {
            for (int t = 0; t < output[b].length; t++) {
                for (int h = 0; h < output[b][t].length; h++) {
                    float value = output[b][t][h];
                    output[b][t][h] = value / (1.0f + (float) Math.exp(-value));
                }
            }
        }
        return output;
    }

    private void validateWeights() {
        validateMatrix(weights.linearStart().kernel(), hiddenSize * 2, hiddenSize, "linearStart");
        validateMatrix(weights.linearEnd().kernel(), hiddenSize, hiddenSize, "linearEnd");
        validateVector(weights.preNormScale(), hiddenSize, "preNormScale");
        validateVector(weights.convolutionNormScale(), hiddenSize, "convolutionNormScale");
        float[][] depthwise = weights.depthwiseKernel();
        if (depthwise.length != hiddenSize) {
            throw new IllegalArgumentException("depthwiseKernel must have " + hiddenSize + " channels");
        }
        for (float[] channel : depthwise) {
            if (channel == null || channel.length != kernelSize) {
                throw new IllegalArgumentException("each depthwise kernel must have length " + kernelSize);
            }
            validateFinite(channel, "depthwiseKernel");
        }
        validateRange(weights.linearStart().clipRange(), "linearStart");
        validateRange(weights.linearEnd().clipRange(), "linearEnd");
    }

    private void validateInput(float[][][] input) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("input must contain at least one batch item");
        }
        int sequence = -1;
        for (float[][] batch : input) {
            if (batch == null || batch.length == 0) {
                throw new IllegalArgumentException("each batch item must contain at least one frame");
            }
            if (sequence < 0) sequence = batch.length;
            if (batch.length != sequence) throw new IllegalArgumentException("input batch must be rectangular");
            for (float[] frame : batch) {
                if (frame == null || frame.length != hiddenSize) {
                    throw new IllegalArgumentException("each frame must have hidden size " + hiddenSize);
                }
                validateFinite(frame, "input");
            }
        }
    }

    private static void validateMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix.length != rows) {
            throw new IllegalArgumentException(name + " output dimension must be " + rows);
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " input dimension must be " + columns);
            }
            validateFinite(row, name);
        }
    }

    private static void validateVector(float[] vector, int size, String name) {
        if (vector.length != size) throw new IllegalArgumentException(name + " length must be " + size);
        validateFinite(vector, name);
    }

    private static void validateFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must contain finite values");
        }
    }

    private static void validateRange(ClipRange range, String name) {
        if (Double.isNaN(range.inputMin()) || Double.isNaN(range.inputMax())
                || Double.isNaN(range.outputMin()) || Double.isNaN(range.outputMax())
                || range.inputMin() > range.inputMax() || range.outputMin() > range.outputMax()) {
            throw new IllegalArgumentException(name + " clipping bounds must be ordered and not NaN");
        }
    }

    private static float clamp(float value, double min, double max) {
        return (float) Math.max(min, Math.min(max, value));
    }

    private static float[][][] clamp(float[][][] input, float limit) {
        float[][][] output = copy(input);
        for (int b = 0; b < output.length; b++) {
            for (int t = 0; t < output[b].length; t++) {
                for (int h = 0; h < output[b][t].length; h++) {
                    output[b][t][h] = clamp(output[b][t][h], -limit, limit);
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

    /** Checkpoint linear weights, with matrix layout [out_features][in_features]. */
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

    /** Input/output clipping bounds read from the checkpoint for a linear layer. */
    public record ClipRange(double inputMin, double inputMax, double outputMin, double outputMax) {
        public static ClipRange unbounded() {
            return new ClipRange(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
    }

    /** All parameters are supplied by a checkpoint mapper; no implicit initialization. */
    public record Weights(
            LinearWeights linearStart,
            LinearWeights linearEnd,
            float[][] depthwiseKernel,
            float[] preNormScale,
            float[] convolutionNormScale) {
        public Weights {
            Objects.requireNonNull(linearStart, "linearStart must not be null");
            Objects.requireNonNull(linearEnd, "linearEnd must not be null");
            depthwiseKernel = copy(depthwiseKernel, "depthwiseKernel");
            preNormScale = Objects.requireNonNull(preNormScale, "preNormScale must not be null").clone();
            convolutionNormScale = Objects.requireNonNull(
                    convolutionNormScale, "convolutionNormScale must not be null").clone();
        }
        @Override public float[][] depthwiseKernel() { return copy(depthwiseKernel, "depthwiseKernel"); }
        @Override public float[] preNormScale() { return preNormScale.clone(); }
        @Override public float[] convolutionNormScale() { return convolutionNormScale.clone(); }

        private static float[][] copy(float[][] source, String name) {
            Objects.requireNonNull(source, name + " must not be null");
            float[][] result = new float[source.length][];
            for (int i = 0; i < source.length; i++) {
                result[i] = source[i] == null ? null : source[i].clone();
            }
            return result;
        }
    }
}
