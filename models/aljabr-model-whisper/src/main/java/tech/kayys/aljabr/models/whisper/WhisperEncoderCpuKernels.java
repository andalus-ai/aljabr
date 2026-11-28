package tech.kayys.aljabr.models.whisper;

import java.util.Objects;

/**
 * Small, deterministic Java CPU reference kernels used to validate Whisper
 * encoder building blocks independently of storage and accelerator providers.
 *
 * <p>All inputs are dense row-major Java arrays. Methods never mutate inputs
 * and allocate their outputs. This is a correctness/reference path, not an
 * optimized production kernel or a complete Whisper model runtime.</p>
 */
public final class WhisperEncoderCpuKernels {
    private WhisperEncoderCpuKernels() {
    }

    /**
     * One-dimensional convolution over channel-first input.
     *
     * @param input [batch, inputChannels, time]
     * @param weights [outputChannels, inputChannels, kernelSize]
     * @param bias [outputChannels]
     * @param stride positive temporal stride
     * @param padding symmetric zero padding
     * @return [batch, outputChannels, outputTime]
     */
    public static float[][][] conv1d(
            float[][][] input, float[][][] weights, float[] bias, int stride, int padding) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(bias, "bias");
        if (stride <= 0 || padding < 0) {
            throw new IllegalArgumentException("stride must be positive and padding non-negative");
        }
        int batch = input.length;
        int inputChannels = requireRectangular3d(input, "input");
        int outputChannels = weights.length;
        requireRectangular3d(weights, "weights");
        int kernelSize = weights[0][0].length;
        if (batch == 0 || inputChannels == 0 || outputChannels == 0 || kernelSize == 0) {
            throw new IllegalArgumentException("convolution dimensions must be non-zero");
        }
        if (bias.length != outputChannels) {
            throw new IllegalArgumentException("bias length must equal output channels");
        }
        int time = input[0][0].length;
        if (time + 2 * padding < kernelSize) {
            throw new IllegalArgumentException("kernel does not fit padded input");
        }
        for (int b = 0; b < batch; b++) {
            if (input[b].length != inputChannels) {
                throw new IllegalArgumentException("input channel count must be consistent");
            }
        }
        for (int out = 0; out < outputChannels; out++) {
            if (weights[out].length != inputChannels) {
                throw new IllegalArgumentException("weight input channel count mismatch");
            }
            for (int in = 0; in < inputChannels; in++) {
                if (weights[out][in].length != kernelSize) {
                    throw new IllegalArgumentException("kernel size must be consistent");
                }
            }
        }
        int outputTime = (time + 2 * padding - kernelSize) / stride + 1;
        float[][][] output = new float[batch][outputChannels][outputTime];
        for (int b = 0; b < batch; b++) {
            for (int out = 0; out < outputChannels; out++) {
                for (int t = 0; t < outputTime; t++) {
                    double sum = bias[out];
                    int inputStart = t * stride - padding;
                    for (int in = 0; in < inputChannels; in++) {
                        for (int k = 0; k < kernelSize; k++) {
                            int inputTime = inputStart + k;
                            if (inputTime >= 0 && inputTime < time) {
                                sum += (double) weights[out][in][k] * input[b][in][inputTime];
                            }
                        }
                    }
                    output[b][out][t] = (float) sum;
                }
            }
        }
        return output;
    }

    /** Applies a dense affine projection to the last input dimension. */
    public static float[][][] linear(
            float[][][] input, float[][] weights, float[] bias) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(bias, "bias");
        int batch = input.length;
        if (batch == 0) throw new IllegalArgumentException("input batch must not be empty");
        int sequence = input[0].length;
        if (sequence == 0) throw new IllegalArgumentException("input sequence must not be empty");
        int inputWidth = input[0][0].length;
        if (inputWidth == 0) throw new IllegalArgumentException("input width must not be empty");
        int outputWidth = weights.length;
        if (outputWidth == 0 || bias.length != outputWidth) {
            throw new IllegalArgumentException("weights and bias output dimensions must match and be non-zero");
        }
        for (int out = 0; out < outputWidth; out++) {
            if (weights[out] == null || weights[out].length != inputWidth) {
                throw new IllegalArgumentException("weight rows must match input width");
            }
        }
        float[][][] output = new float[batch][sequence][outputWidth];
        for (int b = 0; b < batch; b++) {
            if (input[b].length != sequence) throw new IllegalArgumentException("sequence lengths must match");
            for (int t = 0; t < sequence; t++) {
                if (input[b][t].length != inputWidth) throw new IllegalArgumentException("input widths must match");
                for (int out = 0; out < outputWidth; out++) {
                    double sum = bias[out];
                    for (int in = 0; in < inputWidth; in++) {
                        sum += (double) weights[out][in] * input[b][t][in];
                    }
                    output[b][t][out] = (float) sum;
                }
            }
        }
        return output;
    }

    /** Layer normalization over the final dimension, with learned scale and bias. */
    public static float[][][] layerNorm(
            float[][][] input, float[] weight, float[] bias, float epsilon) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(bias, "bias");
        if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        if (input.length == 0 || input[0].length == 0 || input[0][0].length == 0) {
            throw new IllegalArgumentException("input dimensions must be non-zero");
        }
        int batch = input.length;
        int sequence = input[0].length;
        int width = input[0][0].length;
        if (weight.length != width || bias.length != width) {
            throw new IllegalArgumentException("layer norm parameter width mismatch");
        }
        float[][][] output = new float[batch][sequence][width];
        for (int b = 0; b < batch; b++) {
            if (input[b].length != sequence) throw new IllegalArgumentException("sequence lengths must match");
            for (int t = 0; t < sequence; t++) {
                if (input[b][t].length != width) throw new IllegalArgumentException("hidden widths must match");
                double mean = 0.0;
                for (float value : input[b][t]) mean += value;
                mean /= width;
                double variance = 0.0;
                for (float value : input[b][t]) {
                    double centered = value - mean;
                    variance += centered * centered;
                }
                variance /= width;
                double inverseStd = 1.0 / Math.sqrt(variance + epsilon);
                for (int h = 0; h < width; h++) {
                    output[b][t][h] = (float) (((input[b][t][h] - mean) * inverseStd) * weight[h] + bias[h]);
                }
            }
        }
        return output;
    }

    /** Applies the exact-erf form of GELU used by the Whisper encoder. */
    public static float[][][] gelu(float[][][] input) {
        Objects.requireNonNull(input, "input");
        float[][][] output = new float[input.length][][];
        for (int b = 0; b < input.length; b++) {
            output[b] = new float[input[b].length][];
            for (int t = 0; t < input[b].length; t++) {
                output[b][t] = new float[input[b][t].length];
                for (int h = 0; h < input[b][t].length; h++) {
                    double x = input[b][t][h];
                    output[b][t][h] = (float) (0.5 * x * (1.0 + erf(x / Math.sqrt(2.0))));
                }
            }
        }
        return output;
    }

    private static int requireRectangular3d(float[][][] values, String name) {
        if (values.length == 0 || values[0] == null || values[0].length == 0
                || values[0][0] == null || values[0][0].length == 0) {
            throw new IllegalArgumentException(name + " dimensions must be non-zero");
        }
        int channels = values[0].length;
        int time = values[0][0].length;
        for (float[][] sample : values) {
            if (sample == null || sample.length != channels) {
                throw new IllegalArgumentException(name + " channel dimensions must be rectangular");
            }
            for (float[] channel : sample) {
                if (channel == null || channel.length != time) {
                    throw new IllegalArgumentException(name + " temporal dimensions must be rectangular");
                }
            }
        }
        return channels;
    }

    // Abramowitz-Stegun 7.1.26; max absolute error is below 1.5e-7.
    private static double erf(double value) {
        double sign = Math.copySign(1.0, value);
        double x = Math.abs(value);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double polynomial = (((((1.061405429 * t - 1.453152027) * t)
                + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        return sign * (1.0 - polynomial * Math.exp(-x * x));
    }
}
