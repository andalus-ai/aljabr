package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2AttentionBlock;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2EncoderLayer;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfig;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfigResolver;

/**
 * Maps Hugging Face EmbeddingGemma 2 text checkpoint tensor names to the
 * explicit, output-major float32 weight bundle consumed by the reference
 * backbone.
 *
 * <p>This mapper is deliberately independent of Safetensors, GGUF, or any
 * storage API. A format-specific reader supplies named tensors and converts
 * their values to float32; this class validates names/shapes and assembles
 * model-family weights. Non-text modality tensors are ignored, while unknown
 * tensors under {@code language_model.} are rejected to prevent silently
 * incomplete text checkpoints.</p>
 */
public final class EmbeddingGemma2CheckpointMapper {
    private static final String PREFIX = "language_model.";

    private EmbeddingGemma2CheckpointMapper() {
    }

    public static EmbeddingGemma2TextBackbone.Weights map(
            EmbeddingGemma2Config config, Map<String, CheckpointTensor> tensors) {
        Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(tensors, "tensors must not be null");
        Mapping mapping = new Mapping(tensors);

        float[][] tokenEmbeddings = mapping.matrix(
                "language_model.embed_tokens.weight", config.vocabSize(), config.hiddenSize());
        float[][] pleProjection = mapping.matrix(
                "language_model.ple.per_layer_model_projection.weight",
                Math.multiplyExact(config.numHiddenLayers(), config.hiddenSizePerLayerInput()),
                config.hiddenSize());
        float[] pleNormScale = mapping.vector(
                "language_model.ple.per_layer_projection_norm.weight",
                config.hiddenSizePerLayerInput());

        var layers = new java.util.ArrayList<EmbeddingGemma2TextBackbone.LayerWeights>(
                config.numHiddenLayers());
        for (int index = 0; index < config.numHiddenLayers(); index++) {
            String layer = PREFIX + "layers." + index + ".";
            EmbeddingGemma2LayerConfig layerConfig =
                    EmbeddingGemma2LayerConfigResolver.resolve(config, index);
            int hidden = config.hiddenSize();
            int queryWidth = Math.multiplyExact(layerConfig.numAttentionHeads(), layerConfig.headDim());
            int keyValueWidth = Math.multiplyExact(layerConfig.numKeyValueHeads(), layerConfig.headDim());
            int ple = config.hiddenSizePerLayerInput();

            EmbeddingGemma2AttentionBlock.Weights attention =
                    new EmbeddingGemma2AttentionBlock.Weights(
                            mapping.matrix(layer + "self_attn.q_proj.weight", queryWidth, hidden),
                            mapping.optionalVector(layer + "self_attn.q_proj.bias", queryWidth),
                            mapping.matrix(layer + "self_attn.k_proj.weight", keyValueWidth, hidden),
                            mapping.optionalVector(layer + "self_attn.k_proj.bias", keyValueWidth),
                            mapping.matrix(layer + "self_attn.v_proj.weight", keyValueWidth, hidden),
                            mapping.optionalVector(layer + "self_attn.v_proj.bias", keyValueWidth),
                            mapping.matrix(layer + "self_attn.o_proj.weight", hidden, queryWidth),
                            mapping.optionalVector(layer + "self_attn.o_proj.bias", hidden),
                            mapping.vector(layer + "self_attn.q_norm.weight", layerConfig.headDim()),
                            mapping.vector(layer + "self_attn.k_norm.weight", layerConfig.headDim()));

            EmbeddingGemma2Mlp.Weights mlp = new EmbeddingGemma2Mlp.Weights(
                    mapping.matrix(layer + "mlp.gate_proj.weight", config.intermediateSize(), hidden),
                    mapping.matrix(layer + "mlp.up_proj.weight", config.intermediateSize(), hidden),
                    mapping.matrix(layer + "mlp.down_proj.weight", hidden, config.intermediateSize()));

            EmbeddingGemma2EncoderLayer.Weights encoder =
                    new EmbeddingGemma2EncoderLayer.Weights(
                            mapping.vector(layer + "input_layernorm.weight", hidden),
                            mapping.vector(layer + "post_attention_layernorm.weight", hidden),
                            mapping.vector(layer + "pre_feedforward_layernorm.weight", hidden),
                            mapping.vector(layer + "post_feedforward_layernorm.weight", hidden),
                            mapping.matrix(layer + "ple_block.per_layer_input_gate.weight", ple, hidden),
                            mapping.matrix(layer + "ple_block.per_layer_projection.weight", hidden, ple),
                            mapping.vector(layer + "ple_block.post_per_layer_input_norm.weight", hidden));

            // HF persists this non-trainable scalar buffer in some checkpoints.
            // The reference layer currently implements its documented value of one.
            mapping.optionalUnitScalar(layer + "layer_scalar");
            layers.add(new EmbeddingGemma2TextBackbone.LayerWeights(attention, mlp, encoder));
        }

        float[] finalNorm = mapping.vector("language_model.norm.weight", config.hiddenSize());
        float[][] embeddingProjection = mapping.matrix(
                "language_model.embedding_projection.weight", config.embeddingDim(), config.hiddenSize());
        mapping.rejectUnknownTextTensors();

        return new EmbeddingGemma2TextBackbone.Weights(
                tokenEmbeddings, pleProjection, pleNormScale, layers, finalNorm, embeddingProjection);
    }

    /**
     * Dense checkpoint tensor after storage-format decoding and float32 conversion.
     *
     * <p>Use {@link #takeOwnership(int[], float[])} when the caller has freshly
     * allocated arrays and transfers exclusive ownership. This avoids another
     * full copy of very large embedding matrices. Public accessors remain
     * defensive.</p>
     */
    public static final class CheckpointTensor {
        private final int[] shape;
        private final float[] values;

        public CheckpointTensor(int[] shape, float[] values) {
            this(shape, values, false);
        }

        private CheckpointTensor(int[] shape, float[] values, boolean takeOwnership) {
            if (shape == null || values == null) {
                throw new IllegalArgumentException("checkpoint tensor shape and values must not be null");
            }
            this.shape = takeOwnership ? shape : shape.clone();
            this.values = takeOwnership ? values : values.clone();
            long elements = 1L;
            for (int dimension : this.shape) {
                if (dimension <= 0) throw new IllegalArgumentException("tensor dimensions must be positive");
                elements = Math.multiplyExact(elements, dimension);
            }
            if (elements != this.values.length) {
                throw new IllegalArgumentException("tensor shape contains " + elements
                        + " elements but values contains " + this.values.length);
            }
            for (float value : this.values) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("checkpoint tensor values must be finite float32 values");
                }
            }
        }

        /** Transfers exclusive ownership of freshly allocated arrays to this tensor. */
        public static CheckpointTensor takeOwnership(int[] shape, float[] values) {
            return new CheckpointTensor(shape, values, true);
        }

        public int[] shape() { return shape.clone(); }
        public float[] values() { return values.clone(); }
        private float[] valuesForMapping() { return values; }
    }

    private static final class Mapping {
        private final Map<String, CheckpointTensor> tensors;
        private final Set<String> consumed = new HashSet<>();

        private Mapping(Map<String, CheckpointTensor> tensors) {
            this.tensors = tensors;
            for (Map.Entry<String, CheckpointTensor> entry : tensors.entrySet()) {
                if (entry.getKey() == null || entry.getKey().isBlank() || entry.getValue() == null) {
                    throw new IllegalArgumentException("tensor map must contain non-blank names and non-null tensors");
                }
            }
        }

        private float[][] matrix(String name, int rows, int columns) {
            CheckpointTensor tensor = required(name);
            requireShape(name, tensor, rows, columns);
            float[] values = tensor.valuesForMapping();
            float[][] matrix = new float[rows][columns];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(values, row * columns, matrix[row], 0, columns);
            }
            return matrix;
        }

        private float[] vector(String name, int length) {
            CheckpointTensor tensor = required(name);
            requireShape(name, tensor, length);
            return tensor.valuesForMapping();
        }

        private float[] optionalVector(String name, int length) {
            CheckpointTensor tensor = tensors.get(name);
            if (tensor == null) return null;
            consumed.add(name);
            requireShape(name, tensor, length);
            return tensor.values();
        }

        private void optionalUnitScalar(String name) {
            CheckpointTensor tensor = tensors.get(name);
            if (tensor == null) return;
            consumed.add(name);
            int[] shape = tensor.shape();
            if (!(shape.length == 0 || (shape.length == 1 && shape[0] == 1))) {
                throw new IllegalArgumentException(name + " must be a scalar or [1] tensor");
            }
            float[] value = tensor.values();
            if (value.length != 1 || Math.abs(value[0] - 1.0f) > 1.0e-6f) {
                throw new IllegalArgumentException(name + " must equal the reference layer scalar value 1.0");
            }
        }

        private CheckpointTensor required(String name) {
            CheckpointTensor tensor = tensors.get(name);
            if (tensor == null) throw new IllegalArgumentException("missing required checkpoint tensor: " + name);
            consumed.add(name);
            return tensor;
        }

        private void rejectUnknownTextTensors() {
            for (String name : tensors.keySet()) {
                if (name.startsWith(PREFIX) && !consumed.contains(name)) {
                    throw new IllegalArgumentException("unsupported or unmapped text checkpoint tensor: " + name);
                }
            }
        }

        private static void requireShape(String name, CheckpointTensor tensor, int... expected) {
            int[] actual = tensor.shape();
            if (!java.util.Arrays.equals(actual, expected)) {
                throw new IllegalArgumentException(name + " expected shape "
                        + java.util.Arrays.toString(expected) + " but was "
                        + java.util.Arrays.toString(actual));
            }
        }
    }
}
