package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;

/**
 * Maps Hugging Face EmbeddingGemma 2 vision-tower tensors into typed Aljabr
 * reference weights. Storage-format decoding and tensor materialization are
 * deliberately outside this class.
 *
 * <p>The accepted names are the Hugging Face state-dict names rooted at
 * {@code vision_tower.}. Dense matrices are expected in the same
 * output-major layout exposed by the Hugging Face linear modules. The mapper
 * validates the complete supported vision tensor set and rejects unknown
 * tensors in that namespace instead of silently dropping checkpoint data.</p>
 */
public final class EmbeddingGemma2VisionCheckpointMapper {
    private static final String PREFIX = "vision_tower.";

    private EmbeddingGemma2VisionCheckpointMapper() {
    }

    public static Weights map(
            VisionSpec spec,
            Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors) {
        Objects.requireNonNull(spec, "spec must not be null");
        Objects.requireNonNull(tensors, "tensors must not be null");
        Mapping mapping = new Mapping(tensors);

        float[][] patchProjection = mapping.matrix(
                PREFIX + "patch_embedder.input_proj.weight",
                spec.hiddenSize(), Math.multiplyExact(3, Math.multiplyExact(spec.patchSize(), spec.patchSize())));
        float[][][] positionTable = mapping.tensor3(
                PREFIX + "patch_embedder.position_embedding_table",
                2, spec.positionEmbeddingSize(), spec.hiddenSize());

        List<LayerWeights> layers = new ArrayList<>(spec.numHiddenLayers());
        int queryWidth = Math.multiplyExact(spec.numAttentionHeads(), spec.headDim());
        int keyValueWidth = Math.multiplyExact(spec.numKeyValueHeads(), spec.headDim());
        for (int index = 0; index < spec.numHiddenLayers(); index++) {
            String layer = PREFIX + "encoder.layers." + index + ".";
            var attention = new EmbeddingGemma2VisionAttention.Weights(
                    mapping.matrix(layer + "self_attn.q_proj.linear.weight", queryWidth, spec.hiddenSize()),
                    mapping.matrix(layer + "self_attn.k_proj.linear.weight", keyValueWidth, spec.hiddenSize()),
                    mapping.matrix(layer + "self_attn.v_proj.linear.weight", keyValueWidth, spec.hiddenSize()),
                    mapping.matrix(layer + "self_attn.o_proj.linear.weight", spec.hiddenSize(), queryWidth),
                    mapping.vector(layer + "self_attn.q_norm.weight", spec.headDim()),
                    mapping.vector(layer + "self_attn.k_norm.weight", spec.headDim()));

            var mlp = new EmbeddingGemma2Mlp.Weights(
                    mapping.matrix(layer + "mlp.gate_proj.linear.weight", spec.intermediateSize(), spec.hiddenSize()),
                    mapping.matrix(layer + "mlp.up_proj.linear.weight", spec.intermediateSize(), spec.hiddenSize()),
                    mapping.matrix(layer + "mlp.down_proj.linear.weight", spec.hiddenSize(), spec.intermediateSize()));

            var norms = new EmbeddingGemma2VisionEncoderLayer.Weights(
                    mapping.vector(layer + "input_layernorm.weight", spec.hiddenSize()),
                    mapping.vector(layer + "post_attention_layernorm.weight", spec.hiddenSize()),
                    mapping.vector(layer + "pre_feedforward_layernorm.weight", spec.hiddenSize()),
                    mapping.vector(layer + "post_feedforward_layernorm.weight", spec.hiddenSize()));
            layers.add(new LayerWeights(attention, mlp, norms));
        }

        // Some converted Orbax checkpoints carry this legacy tensor, although
        // the current Hugging Face Gemma4VisionPooler forward path does not use it.
        // Accept and retain it when present; never require it for HF state dicts.
        float[] legacyPoolerScale = mapping.optionalVectorOrExpanded(
                PREFIX + "pooler.scale", spec.hiddenSize(), new int[]{1, 1, spec.hiddenSize()});

        float[] stdBias = null;
        float[] stdScale = null;
        if (spec.standardize()) {
            stdBias = mapping.vector(PREFIX + "std_bias", spec.hiddenSize());
            stdScale = mapping.vector(PREFIX + "std_scale", spec.hiddenSize());
        } else {
            mapping.optionalVector(PREFIX + "std_bias", spec.hiddenSize());
            mapping.optionalVector(PREFIX + "std_scale", spec.hiddenSize());
        }

        mapping.rejectUnknownVisionTensors();
        return new Weights(
                new EmbeddingGemma2VisionPatchEmbedder.Weights(patchProjection, positionTable),
                layers, legacyPoolerScale, stdBias, stdScale);
    }

    /** Configuration values required to validate the vision checkpoint. */
    public record VisionSpec(
            int hiddenSize,
            int intermediateSize,
            int numHiddenLayers,
            int numAttentionHeads,
            int numKeyValueHeads,
            int headDim,
            int patchSize,
            int positionEmbeddingSize,
            boolean standardize) {
        public VisionSpec {
            if (hiddenSize <= 0 || intermediateSize <= 0 || numHiddenLayers <= 0
                    || numAttentionHeads <= 0 || numKeyValueHeads <= 0 || headDim <= 0
                    || patchSize <= 0 || positionEmbeddingSize <= 0) {
                throw new IllegalArgumentException("vision dimensions must be positive");
            }
            if (numAttentionHeads % numKeyValueHeads != 0) {
                throw new IllegalArgumentException("numAttentionHeads must be divisible by numKeyValueHeads");
            }
            if (Math.multiplyExact(numAttentionHeads, headDim) != hiddenSize) {
                throw new IllegalArgumentException("query head width must equal hiddenSize");
            }
        }
    }

    /** All checkpoint-backed weights needed by the current vision reference components. */
    public record Weights(
            EmbeddingGemma2VisionPatchEmbedder.Weights patchEmbedder,
            List<LayerWeights> layers,
            float[] legacyPoolerScale,
            float[] standardizationBias,
            float[] standardizationScale) {
        public Weights {
            Objects.requireNonNull(patchEmbedder, "patchEmbedder must not be null");
            layers = List.copyOf(Objects.requireNonNull(layers, "layers must not be null"));
            legacyPoolerScale = legacyPoolerScale == null ? null : legacyPoolerScale.clone();
            standardizationBias = standardizationBias == null ? null : standardizationBias.clone();
            standardizationScale = standardizationScale == null ? null : standardizationScale.clone();
            if ((standardizationBias == null) != (standardizationScale == null)) {
                throw new IllegalArgumentException("standardization bias and scale must both be present or absent");
            }
        }
    }

    public record LayerWeights(
            EmbeddingGemma2VisionAttention.Weights attention,
            EmbeddingGemma2Mlp.Weights mlp,
            EmbeddingGemma2VisionEncoderLayer.Weights norms) {
        public LayerWeights {
            Objects.requireNonNull(attention, "attention must not be null");
            Objects.requireNonNull(mlp, "mlp must not be null");
            Objects.requireNonNull(norms, "norms must not be null");
        }
    }

    private static final class Mapping {
        private final Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors;
        private final Set<String> consumed = new HashSet<>();

        private Mapping(Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors) {
            this.tensors = tensors;
            for (Map.Entry<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> entry : tensors.entrySet()) {
                if (entry.getKey() == null || entry.getKey().isBlank() || entry.getValue() == null) {
                    throw new IllegalArgumentException("tensor map must contain non-blank names and non-null tensors");
                }
            }
        }

        private float[][] matrix(String name, int rows, int columns) {
            var tensor = required(name);
            requireShape(name, tensor, rows, columns);
            float[] values = tensor.values();
            float[][] matrix = new float[rows][columns];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(values, row * columns, matrix[row], 0, columns);
            }
            return matrix;
        }

        private float[][][] tensor3(String name, int first, int second, int third) {
            var tensor = required(name);
            requireShape(name, tensor, first, second, third);
            float[] values = tensor.values();
            float[][][] result = new float[first][second][third];
            int offset = 0;
            for (int i = 0; i < first; i++) {
                for (int j = 0; j < second; j++) {
                    System.arraycopy(values, offset, result[i][j], 0, third);
                    offset += third;
                }
            }
            return result;
        }

        private float[] vector(String name, int length) {
            var tensor = required(name);
            requireShape(name, tensor, length);
            return tensor.values();
        }

        private float[] optionalVectorOrExpanded(String name, int length, int[] expandedShape) {
            var tensor = tensors.get(name);
            if (tensor == null) return null;
            int[] shape = tensor.shape();
            if (!java.util.Arrays.equals(shape, new int[]{length})
                    && !java.util.Arrays.equals(shape, expandedShape)) {
                throw new IllegalArgumentException(name + " expected shape [" + length + "] or "
                        + java.util.Arrays.toString(expandedShape) + " but was "
                        + java.util.Arrays.toString(shape));
            }
            consumed.add(name);
            return tensor.values();
        }

        private void optionalVector(String name, int length) {
            var tensor = tensors.get(name);
            if (tensor == null) return;
            consumed.add(name);
            requireShape(name, tensor, length);
        }

        private EmbeddingGemma2CheckpointMapper.CheckpointTensor required(String name) {
            var tensor = tensors.get(name);
            if (tensor == null) {
                throw new IllegalArgumentException("missing required vision checkpoint tensor: " + name);
            }
            consumed.add(name);
            return tensor;
        }

        private void rejectUnknownVisionTensors() {
            for (String name : tensors.keySet()) {
                if (name.startsWith(PREFIX) && !consumed.contains(name)) {
                    throw new IllegalArgumentException("unsupported or unmapped vision checkpoint tensor: " + name);
                }
            }
        }

        private static void requireShape(
                String name, EmbeddingGemma2CheckpointMapper.CheckpointTensor tensor, int... expected) {
            int[] actual = tensor.shape();
            if (!java.util.Arrays.equals(actual, expected)) {
                throw new IllegalArgumentException(name + " expected shape "
                        + java.util.Arrays.toString(expected) + " but was "
                        + java.util.Arrays.toString(actual));
            }
        }
    }
}
