package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Reference implementation of Gemma 4's two-stage audio subsampling
 * convolutional projection (SSCP).
 *
 * <p>Input features use {@code [batch][time][mel]}. The two checkpoint-backed
 * Conv2d layers use NCHW internally, 3x3 kernels, stride two, padding one,
 * no convolution bias, channel-wise LayerNorm without bias, and ReLU. The
 * final map is flattened in frequency-major/channel-minor order and projected
 * to the audio encoder hidden size.</p>
 *
 * <p>This class is deliberately limited to the convolutional projection. It
 * does not implement the subsequent Conformer stack or audio output
 * projection. Weight arrays are supplied by the checkpoint loader; no
 * parameters are randomly initialized here.</p>
 */
public final class EmbeddingGemma2AudioSubsampling {
    private static final int KERNEL = 3;
    private static final int STRIDE = 2;
    private static final int PADDING = 1;

    private final int inputFeatures;
    private final int hiddenSize;
    private final int firstChannels;
    private final int secondChannels;
    private final float epsilon;
    private final Weights weights;

    public EmbeddingGemma2AudioSubsampling(EmbeddingGemma2AudioConfig config, Weights weights) {
        Objects.requireNonNull(config, "config must not be null");
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        if (config.subsamplingConvolutionChannels().size() != 2) {
            throw new IllegalArgumentException("Gemma4 audio subsampling requires exactly two convolution channel sizes");
        }
        this.inputFeatures = config.featureSize();
        this.hiddenSize = config.hiddenSize();
        this.firstChannels = config.subsamplingConvolutionChannels().get(0);
        this.secondChannels = config.subsamplingConvolutionChannels().get(1);
        this.epsilon = (float) config.rmsNormEps();
        validateWeights();
    }

    /**
     * Applies SSCP to a batch of log-mel features.
     *
     * @param inputFeatures log-mel values shaped [batch][time][feature]
     * @param inputFeaturesMask optional true=valid frame mask [batch][time]
     * @return projected hidden states [batch][ceil(time/4)][hiddenSize] and
     *         the mask downsampled by taking every second element twice
     */
    public Output compute(float[][][] inputFeatures, boolean[][] inputFeaturesMask) {
        validateInput(inputFeatures, inputFeaturesMask);
        int batch = inputFeatures.length;
        int time = inputFeatures[0].length;

        float[][][][] hidden = new float[batch][1][time][this.inputFeatures];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < time; t++) {
                if (inputFeaturesMask == null || inputFeaturesMask[b][t]) {
                    System.arraycopy(inputFeatures[b][t], 0, hidden[b][0][t], 0, this.inputFeatures);
                }
            }
        }

        boolean[][] mask = copyMask(inputFeaturesMask);
        hidden = applyLayer(hidden, weights.layer0(), 1, firstChannels);
        mask = downsampleMask(mask);
        hidden = applyLayer(hidden, weights.layer1(), firstChannels, secondChannels);
        mask = downsampleMask(mask);

        int outputTime = hidden[0][0].length;
        int outputFrequency = hidden[0][0][0].length;
        int projectionWidth = Math.multiplyExact(outputFrequency, secondChannels);
        float[][][] projected = new float[batch][outputTime][hiddenSize];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < outputTime; t++) {
                int offset = 0;
                for (int f = 0; f < outputFrequency; f++) {
                    for (int c = 0; c < secondChannels; c++) {
                        float value = hidden[b][c][t][f];
                        for (int h = 0; h < hiddenSize; h++) {
                            projected[b][t][h] += value * weights.inputProjection()[h][offset];
                        }
                        offset++;
                    }
                }
            }
        }
        return new Output(projected, mask, outputFrequency);
    }

    private float[][][][] applyLayer(
            float[][][][] input, ConvLayerWeights layer, int inputChannels, int outputChannels) {
        int batch = input.length;
        int inputTime = input[0][0].length;
        int inputFrequency = input[0][0][0].length;
        int outputTime = outputSize(inputTime);
        int outputFrequency = outputSize(inputFrequency);
        float[][][][] convolved = new float[batch][outputChannels][outputTime][outputFrequency];

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outputChannels; oc++) {
                for (int oy = 0; oy < outputTime; oy++) {
                    for (int ox = 0; ox < outputFrequency; ox++) {
                        float sum = 0f;
                        for (int ic = 0; ic < inputChannels; ic++) {
                            for (int ky = 0; ky < KERNEL; ky++) {
                                int iy = oy * STRIDE + ky - PADDING;
                                if (iy < 0 || iy >= inputTime) continue;
                                for (int kx = 0; kx < KERNEL; kx++) {
                                    int ix = ox * STRIDE + kx - PADDING;
                                    if (ix >= 0 && ix < inputFrequency) {
                                        sum += input[b][ic][iy][ix] * layer.kernel()[oc][ic][ky][kx];
                                    }
                                }
                            }
                        }
                        convolved[b][oc][oy][ox] = sum;
                    }
                }
            }
        }
        return normalizeAndActivate(convolved, layer.layerNormScale());
    }

    /** LayerNorm is applied over channels at each time/frequency position, then ReLU. */
    private float[][][][] normalizeAndActivate(float[][][][] values, float[] scale) {
        int batch = values.length;
        int channels = values[0].length;
        int time = values[0][0].length;
        int frequency = values[0][0][0].length;
        float[][][][] output = new float[batch][channels][time][frequency];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < frequency; f++) {
                    float mean = 0f;
                    for (int c = 0; c < channels; c++) mean += values[b][c][t][f];
                    mean /= channels;
                    float variance = 0f;
                    for (int c = 0; c < channels; c++) {
                        float centered = values[b][c][t][f] - mean;
                        variance += centered * centered;
                    }
                    variance /= channels;
                    float inverseStd = (float) (1.0 / Math.sqrt(variance + epsilon));
                    for (int c = 0; c < channels; c++) {
                        float normalized = (values[b][c][t][f] - mean) * inverseStd * scale[c];
                        output[b][c][t][f] = Math.max(0f, normalized);
                    }
                }
            }
        }
        return output;
    }

    private void validateWeights() {
        validateKernel(weights.layer0(), firstChannels, 1, "layer0");
        validateKernel(weights.layer1(), secondChannels, firstChannels, "layer1");
        int projectedFrequency = outputSize(outputSize(inputFeatures));
        int projectionWidth = Math.multiplyExact(projectedFrequency, secondChannels);
        if (weights.inputProjection().length != hiddenSize) {
            throw new IllegalArgumentException("inputProjection must have " + hiddenSize + " output rows");
        }
        for (float[] row : weights.inputProjection()) {
            if (row == null || row.length != projectionWidth) {
                throw new IllegalArgumentException("inputProjection rows must have width " + projectionWidth);
            }
            requireFinite(row, "inputProjection");
        }
    }

    private static void validateKernel(ConvLayerWeights layer, int outputChannels, int inputChannels, String name) {
        Objects.requireNonNull(layer, name + " weights must not be null");
        if (layer.kernel().length != outputChannels || layer.layerNormScale().length != outputChannels) {
            throw new IllegalArgumentException(name + " output channel dimensions do not match config");
        }
        for (int oc = 0; oc < outputChannels; oc++) {
            if (layer.kernel()[oc] == null || layer.kernel()[oc].length != inputChannels) {
                throw new IllegalArgumentException(name + " kernel input channels do not match config");
            }
            for (int ic = 0; ic < inputChannels; ic++) {
                if (layer.kernel()[oc][ic] == null || layer.kernel()[oc][ic].length != KERNEL) {
                    throw new IllegalArgumentException(name + " kernel height must be 3");
                }
                for (float[] row : layer.kernel()[oc][ic]) {
                    if (row == null || row.length != KERNEL) {
                        throw new IllegalArgumentException(name + " kernel width must be 3");
                    }
                    requireFinite(row, name + " kernel");
                }
            }
        }
        requireFinite(layer.layerNormScale(), name + " layerNormScale");
    }

    private void validateInput(float[][][] input, boolean[][] mask) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("inputFeatures must contain at least one batch item");
        }
        if (mask != null && mask.length != input.length) {
            throw new IllegalArgumentException("inputFeaturesMask batch dimension must match inputFeatures");
        }
        int time = -1;
        for (int b = 0; b < input.length; b++) {
            if (input[b] == null || input[b].length == 0) {
                throw new IllegalArgumentException("each audio item must contain at least one frame");
            }
            if (time < 0) time = input[b].length;
            if (input[b].length != time) {
                throw new IllegalArgumentException("inputFeatures must be a rectangular batch");
            }
            for (float[] frame : input[b]) {
                if (frame == null || frame.length != inputFeatures) {
                    throw new IllegalArgumentException("each frame must have feature width " + inputFeatures);
                }
                requireFinite(frame, "inputFeatures");
            }
            if (mask != null && (mask.length != input.length || mask[b] == null || mask[b].length != time)) {
                throw new IllegalArgumentException("inputFeaturesMask must have shape [batch][time]");
            }
        }
    }

    private static boolean[][] downsampleMask(boolean[][] mask) {
        if (mask == null) return null;
        boolean[][] result = new boolean[mask.length][];
        for (int b = 0; b < mask.length; b++) {
            result[b] = new boolean[(mask[b].length + 1) / 2];
            for (int i = 0; i < result[b].length; i++) result[b][i] = mask[b][i * 2];
        }
        return result;
    }

    private static boolean[][] copyMask(boolean[][] mask) {
        if (mask == null) return null;
        boolean[][] copy = new boolean[mask.length][];
        for (int i = 0; i < mask.length; i++) copy[i] = mask[i].clone();
        return copy;
    }

    private static int outputSize(int inputSize) {
        return (inputSize + 1) / 2;
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must contain finite values");
        }
    }


    /** One convolution's checkpoint parameters; kernel layout is [out][in][3][3]. */
    public record ConvLayerWeights(float[][][][] kernel, float[] layerNormScale) {
        public ConvLayerWeights {
            Objects.requireNonNull(kernel, "kernel must not be null");
            Objects.requireNonNull(layerNormScale, "layerNormScale must not be null");
        }
    }

    /** All SSCP checkpoint parameters. Projection layout is [hidden][flattenedFrequencyAndChannel]. */
    public record Weights(ConvLayerWeights layer0, ConvLayerWeights layer1, float[][] inputProjection) {
        public Weights {
            Objects.requireNonNull(layer0, "layer0 must not be null");
            Objects.requireNonNull(layer1, "layer1 must not be null");
            Objects.requireNonNull(inputProjection, "inputProjection must not be null");
        }
    }

    /** Immutable output; tensor-shaped arrays and optional mask are copied on ingress and egress. */
    public record Output(float[][][] hiddenStates, boolean[][] mask, int frequencySize) {
        public Output {
            Objects.requireNonNull(hiddenStates, "hiddenStates must not be null");
            float[][][] copy = new float[hiddenStates.length][][];
            for (int b = 0; b < copy.length; b++) {
                copy[b] = new float[hiddenStates[b].length][];
                for (int t = 0; t < copy[b].length; t++) copy[b][t] = hiddenStates[b][t].clone();
            }
            hiddenStates = copy;
            mask = copyMask(mask);
        }
        @Override public float[][][] hiddenStates() {
            float[][][] copy = new float[hiddenStates.length][][];
            for (int b = 0; b < copy.length; b++) {
                copy[b] = new float[hiddenStates[b].length][];
                for (int t = 0; t < copy[b].length; t++) copy[b][t] = hiddenStates[b][t].clone();
            }
            return copy;
        }
        @Override public boolean[][] mask() { return copyMask(mask); }
    }
}
