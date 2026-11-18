package tech.kayys.aljabr.models.embeddinggemma2.attention;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfig;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfigResolver;
import tech.kayys.aljabr.models.embeddinggemma2.ple.EmbeddingGemma2PerLayerEmbeddingProjector;

/**
 * Composes the complete EmbeddingGemma 2 text transformer backbone.
 *
 * <p>This is a float32 reference path from already-scaled token embeddings
 * through PLE projection, every encoder layer, final RMSNorm, and the learned
 * token-wise embedding projection. Tokenization, vocabulary lookup, checkpoint
 * deserialization, mean pooling, and vector post-processing remain explicit
 * outer stages.</p>
 *
 * <p>Every parameter is supplied by the checkpoint. No tensor is initialized
 * or synthesized by this class.</p>
 */
public final class EmbeddingGemma2TextBackbone {

    private final EmbeddingGemma2Config config;
    private final float[][] pleProjectionWeights;
    private final float[] pleNormScale;
    private final List<EmbeddingGemma2EncoderLayer> layers;
    private final float[] finalNormScale;
    private final float[][] embeddingProjectionWeights;

    public EmbeddingGemma2TextBackbone(EmbeddingGemma2Config config, Weights weights) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(weights, "weights must not be null");
        int hidden = config.hiddenSize();
        int ple = config.hiddenSizePerLayerInput();
        int layerCount = config.numHiddenLayers();

        requireMatrix(weights.pleProjection(), Math.multiplyExact(layerCount, ple), hidden,
                "pleProjection");
        requireVector(weights.pleNormScale(), ple, "pleNormScale");
        requireVector(weights.finalNormScale(), hidden, "finalNormScale");
        requireMatrix(weights.embeddingProjection(), config.embeddingDim(), hidden,
                "embeddingProjection");
        if (weights.layers() == null || weights.layers().size() != layerCount) {
            throw new IllegalArgumentException("exactly one checkpoint layer weight bundle is required per layer");
        }

        this.pleProjectionWeights = copy(weights.pleProjection());
        this.pleNormScale = weights.pleNormScale().clone();
        this.finalNormScale = weights.finalNormScale().clone();
        this.embeddingProjectionWeights = copy(weights.embeddingProjection());

        List<EmbeddingGemma2EncoderLayer> built = new ArrayList<>(layerCount);
        for (int i = 0; i < layerCount; i++) {
            LayerWeights layerWeights = Objects.requireNonNull(weights.layers().get(i),
                    "layer weights must not be null at index " + i);
            EmbeddingGemma2LayerConfig layerConfig =
                    EmbeddingGemma2LayerConfigResolver.resolve(config, i);
            built.add(new EmbeddingGemma2EncoderLayer(config, layerConfig,
                    layerWeights.attention(), layerWeights.mlp(), layerWeights.encoder()));
        }
        this.layers = List.copyOf(built);
    }

    /**
     * Executes the text backbone.
     *
     * @param scaledTokenEmbeddings checkpoint vocabulary rows after Gemma's
     *        embedding scale, shape [sequence][hiddenSize]
     * @param positionIds explicit position ids, shape [sequence]
     * @param validKeys optional true=valid key mask, shape [sequence]
     * @return token-level projected states [sequence][embeddingDim]
     */
    public float[][] encode(float[][] scaledTokenEmbeddings, int[] positionIds, boolean[] validKeys) {
        validateSequence(scaledTokenEmbeddings, config.hiddenSize(), "scaledTokenEmbeddings");
        int sequence = scaledTokenEmbeddings.length;
        if (positionIds == null || positionIds.length != sequence) {
            throw new IllegalArgumentException("positionIds length must equal sequence length");
        }
        if (validKeys != null && validKeys.length != sequence) {
            throw new IllegalArgumentException("validKeys length must equal sequence length");
        }

        float[][][] perLayerInputs = EmbeddingGemma2PerLayerEmbeddingProjector.project(
                scaledTokenEmbeddings, pleProjectionWeights, pleNormScale,
                config.numHiddenLayers(), config.hiddenSizePerLayerInput(),
                (float) config.rmsNormEps());

        float[][] hiddenStates = copy(scaledTokenEmbeddings);
        for (int layer = 0; layer < layers.size(); layer++) {
            float[][] layerPle = new float[sequence][config.hiddenSizePerLayerInput()];
            for (int token = 0; token < sequence; token++) {
                layerPle[token] = perLayerInputs[token][layer].clone();
            }
            hiddenStates = layers.get(layer).compute(hiddenStates, layerPle, positionIds, validKeys);
        }

        float[][] normalized = EmbeddingGemma2SequenceRmsNorm.apply(
                hiddenStates, finalNormScale, (float) config.rmsNormEps());
        return EmbeddingGemma2LinearProjection.apply(normalized, embeddingProjectionWeights, null);
    }

    public int hiddenSize() { return config.hiddenSize(); }
    public int embeddingDimension() { return config.embeddingDim(); }
    public int layerCount() { return layers.size(); }

    private static void validateSequence(float[][] values, int width, String name) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException(name + " must be non-empty");
        }
        for (float[] row : values) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException(name + " must have feature width " + width);
            }
            for (float value : row) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException(name + " must contain finite values");
                }
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
        for (float value : vector) if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(name + " must contain finite values");
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
            for (float value : row) if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain finite values");
            }
        }
    }

    private static float[][] copy(float[][] values) {
        float[][] result = new float[values.length][];
        for (int i = 0; i < values.length; i++) result[i] = values[i].clone();
        return result;
    }

    /** Immutable-by-convention checkpoint bundle for the entire text backbone. */
    public record Weights(
            float[][] pleProjection,
            float[] pleNormScale,
            List<LayerWeights> layers,
            float[] finalNormScale,
            float[][] embeddingProjection) {
        public Weights {
            layers = layers == null ? null : List.copyOf(layers);
        }
    }

    /** Checkpoint tensors for one transformer block. */
    public record LayerWeights(
            EmbeddingGemma2AttentionBlock.Weights attention,
            EmbeddingGemma2Mlp.Weights mlp,
            EmbeddingGemma2EncoderLayer.Weights encoder) {
    }
}
