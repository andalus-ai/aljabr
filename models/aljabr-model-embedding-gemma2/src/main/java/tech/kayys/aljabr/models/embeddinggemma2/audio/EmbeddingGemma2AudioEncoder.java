package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Complete Java-float reference implementation of the Gemma 4 audio tower
 * embedded in an EmbeddingGemma 2 checkpoint.
 *
 * <p>The forward path is subsampling convolutional projection, the configured
 * number of Conformer audio layers, and the final biased output projection.
 * The returned mask is the subsampling mask and uses {@code true = valid}.</p>
 *
 * <p>All parameters are supplied by the caller/checkpoint mapper. This class
 * does not initialize weights, read checkpoint files, or claim accelerated
 * Aljabr Tensor/backend execution.</p>
 */
public final class EmbeddingGemma2AudioEncoder {
    private final EmbeddingGemma2AudioConfig config;
    private final EmbeddingGemma2AudioSubsampling subsampling;
    private final List<EmbeddingGemma2AudioConformerLayer> layers;
    private final float[][] outputProjection;
    private final float[] outputProjectionBias;

    public EmbeddingGemma2AudioEncoder(
            EmbeddingGemma2AudioConfig config,
            Weights weights) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(weights, "weights must not be null");

        this.subsampling = new EmbeddingGemma2AudioSubsampling(config, weights.subsampling());
        if (weights.layers().size() != config.numHiddenLayers()) {
            throw new IllegalArgumentException(
                    "audio layer weights count must equal numHiddenLayers ("
                            + config.numHiddenLayers() + ")");
        }

        List<EmbeddingGemma2AudioConformerLayer> constructed =
                new ArrayList<>(config.numHiddenLayers());
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            constructed.add(new EmbeddingGemma2AudioConformerLayer(config, weights.layers().get(i)));
        }
        this.layers = List.copyOf(constructed);
        this.outputProjection = copyAndValidateMatrix(
                weights.outputProjection(), config.outputProjectionSize(),
                config.hiddenSize(), "outputProjection");
        this.outputProjectionBias = copyAndValidateVector(
                weights.outputProjectionBias(), config.outputProjectionSize(),
                "outputProjectionBias");
    }

    /**
     * Encodes log-mel features.
     *
     * @param inputFeatures log-mel features [batch][time][featureSize]
     * @param inputMask optional true=valid input frame mask [batch][time]
     * @return projected audio soft-token states and the downsampled true=valid mask
     */
    public Output compute(float[][][] inputFeatures, boolean[][] inputMask) {
        EmbeddingGemma2AudioSubsampling.Output projected =
                subsampling.compute(inputFeatures, inputMask);
        float[][][] hiddenStates = projected.hiddenStates();
        boolean[][] outputMask = projected.mask();

        for (EmbeddingGemma2AudioConformerLayer layer : layers) {
            hiddenStates = layer.compute(hiddenStates, outputMask);
        }

        return new Output(applyOutputProjection(hiddenStates), outputMask);
    }

    private float[][][] applyOutputProjection(float[][][] hiddenStates) {
        float[][][] output = new float[hiddenStates.length]
                [hiddenStates[0].length][config.outputProjectionSize()];
        for (int b = 0; b < hiddenStates.length; b++) {
            for (int t = 0; t < hiddenStates[b].length; t++) {
                for (int out = 0; out < outputProjection.length; out++) {
                    double sum = outputProjectionBias[out];
                    for (int in = 0; in < config.hiddenSize(); in++) {
                        sum += (double) outputProjection[out][in] * hiddenStates[b][t][in];
                    }
                    output[b][t][out] = (float) sum;
                }
            }
        }
        return output;
    }

    private static float[][] copyAndValidateMatrix(
            float[][] matrix, int rows, int columns, String name) {
        Objects.requireNonNull(matrix, name + " must not be null");
        if (matrix.length != rows) {
            throw new IllegalArgumentException(name + " must have " + rows + " rows");
        }
        float[][] copy = new float[rows][];
        for (int row = 0; row < rows; row++) {
            if (matrix[row] == null || matrix[row].length != columns) {
                throw new IllegalArgumentException(name + " rows must have " + columns + " columns");
            }
            copy[row] = matrix[row].clone();
            requireFinite(copy[row], name);
        }
        return copy;
    }

    private static float[] copyAndValidateVector(float[] vector, int size, String name) {
        Objects.requireNonNull(vector, name + " must not be null");
        if (vector.length != size) {
            throw new IllegalArgumentException(name + " must have length " + size);
        }
        float[] copy = vector.clone();
        requireFinite(copy, name);
        return copy;
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain only finite values");
            }
        }
    }

    /** All checkpoint-backed parameters for the complete audio tower. */
    public record Weights(
            EmbeddingGemma2AudioSubsampling.Weights subsampling,
            List<EmbeddingGemma2AudioConformerLayer.Weights> layers,
            float[][] outputProjection,
            float[] outputProjectionBias) {
        public Weights {
            Objects.requireNonNull(subsampling, "subsampling must not be null");
            Objects.requireNonNull(layers, "layers must not be null");
            layers = List.copyOf(layers);
            if (layers.stream().anyMatch(Objects::isNull)) {
                throw new IllegalArgumentException("layers must not contain null entries");
            }
            Objects.requireNonNull(outputProjection, "outputProjection must not be null");
            Objects.requireNonNull(outputProjectionBias, "outputProjectionBias must not be null");
        }
    }

    /** Audio tower result. Arrays are defensively copied at both boundaries. */
    public record Output(float[][][] lastHiddenState, boolean[][] attentionMask) {
        public Output {
            Objects.requireNonNull(lastHiddenState, "lastHiddenState must not be null");
            lastHiddenState = copy3d(lastHiddenState);
            attentionMask = copy2d(attentionMask);
        }

        @Override
        public float[][][] lastHiddenState() {
            return copy3d(lastHiddenState);
        }

        @Override
        public boolean[][] attentionMask() {
            return copy2d(attentionMask);
        }

        private static float[][][] copy3d(float[][][] source) {
            float[][][] copy = new float[source.length][][];
            for (int b = 0; b < source.length; b++) {
                copy[b] = new float[source[b].length][];
                for (int t = 0; t < source[b].length; t++) {
                    copy[b][t] = source[b][t].clone();
                }
            }
            return copy;
        }

        private static boolean[][] copy2d(boolean[][] source) {
            if (source == null) return null;
            boolean[][] copy = new boolean[source.length][];
            for (int b = 0; b < source.length; b++) {
                copy[b] = source[b].clone();
            }
            return copy;
        }
    }
}
