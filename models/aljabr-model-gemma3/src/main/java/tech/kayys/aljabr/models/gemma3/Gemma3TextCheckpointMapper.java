package tech.kayys.aljabr.models.gemma3;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.spi.model.ModelConfig;

/**
 * Maps named Hugging Face Gemma 3 text checkpoint tensors into an immutable
 * float32 weight bundle independent of the checkpoint storage format.
 *
 * <p>Format adapters decode tensors and call {@link #map(ModelConfig,
 * Gemma3TextFamily, Map)}. The mapper owns architecture naming, shape
 * validation, and the canonical output-major weight layout; it does not
 * perform file I/O or execute the model.</p>
 */
public final class Gemma3TextCheckpointMapper {
    private Gemma3TextCheckpointMapper() {
    }

    public static Weights map(
            ModelConfig config,
            Gemma3TextFamily architecture,
            Map<String, CheckpointTensor> tensors) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(architecture, "architecture");
        Objects.requireNonNull(tensors, "tensors");
        if (config.numHiddenLayers() <= 0 || config.hiddenSize() <= 0
                || config.intermediateSize() <= 0 || config.vocabSize() <= 0) {
            throw new IllegalArgumentException("Gemma 3 dimensions must be positive");
        }

        Mapping mapping = new Mapping(tensors);
        int hidden = config.hiddenSize();
        int headDim = config.resolvedHeadDim();
        int queryWidth = Math.multiplyExact(config.numAttentionHeads(), headDim);
        int kvWidth = Math.multiplyExact(config.resolvedNumKvHeads(), headDim);

        float[][] embeddings = mapping.matrix(
                architecture.embedTokensWeight(), config.vocabSize(), hidden);
        float[] finalNorm = mapping.vector(architecture.finalNormWeight(), hidden);

        List<LayerWeights> layers = new ArrayList<>(config.numHiddenLayers());
        for (int index = 0; index < config.numHiddenLayers(); index++) {
            String query = architecture.layerQueryWeight(index);
            String key = architecture.layerKeyWeight(index);
            String value = architecture.layerValueWeight(index);
            String output = architecture.layerOutputWeight(index);
            String gate = architecture.layerFfnGateWeight(index);
            String up = architecture.layerFfnUpWeight(index);
            String down = architecture.layerFfnDownWeight(index);

            layers.add(new LayerWeights(
                    mapping.matrix(query, queryWidth, hidden),
                    mapping.matrix(key, kvWidth, hidden),
                    mapping.matrix(value, kvWidth, hidden),
                    mapping.matrix(output, hidden, queryWidth),
                    mapping.vector(architecture.layerAttentionNormWeight(index), hidden),
                    mapping.vector(architecture.layerPostAttnNormWeight(index), hidden),
                    mapping.vector(architecture.layerPreFfnNormWeight(index), hidden),
                    mapping.vector(architecture.layerPostFfnNormWeight(index), hidden),
                    mapping.vector(architecture.layerQueryNormWeight(index), headDim),
                    mapping.vector(architecture.layerKeyNormWeight(index), headDim),
                    mapping.matrix(gate, config.intermediateSize(), hidden),
                    mapping.matrix(up, config.intermediateSize(), hidden),
                    mapping.matrix(down, hidden, config.intermediateSize())));
        }

        float[][] lmHead = architecture.hasTiedEmbeddings()
                ? embeddings
                : mapping.matrix(architecture.lmHeadWeight(), config.vocabSize(), hidden);
        mapping.rejectUnconsumedModelWeights();
        return new Weights(embeddings, finalNorm, lmHead, layers);
    }

    /** Detached, output-major float32 checkpoint tensor. */
    public static final class CheckpointTensor {
        private final int[] shape;
        private final float[] values;

        public CheckpointTensor(int[] shape, float[] values) {
            Objects.requireNonNull(shape, "shape");
            Objects.requireNonNull(values, "values");
            this.shape = shape.clone();
            this.values = values.clone();
            long count = 1;
            for (int dimension : this.shape) {
                if (dimension <= 0) throw new IllegalArgumentException("tensor dimensions must be positive");
                count = Math.multiplyExact(count, dimension);
            }
            if (count != this.values.length) {
                throw new IllegalArgumentException("tensor shape has " + count
                        + " elements but values has " + this.values.length);
            }
            for (float value : this.values) {
                if (!Float.isFinite(value)) throw new IllegalArgumentException("tensor values must be finite");
            }
        }

        public int[] shape() { return shape.clone(); }
        public float[] values() { return values.clone(); }
    }

    public record LayerWeights(
            float[][] query,
            float[][] key,
            float[][] value,
            float[][] attentionOutput,
            float[] attentionNorm,
            float[] postAttentionNorm,
            float[] preFeedForwardNorm,
            float[] postFeedForwardNorm,
            float[] queryNorm,
            float[] keyNorm,
            float[][] feedForwardGate,
            float[][] feedForwardUp,
            float[][] feedForwardDown) {
    }

    public record Weights(
            float[][] tokenEmbeddings,
            float[] finalNorm,
            float[][] languageModelHead,
            List<LayerWeights> layers) {
        public Weights {
            layers = List.copyOf(layers);
        }
    }

    private static final class Mapping {
        private final Map<String, CheckpointTensor> tensors;
        private final Set<String> consumed = new HashSet<>();

        private Mapping(Map<String, CheckpointTensor> tensors) {
            this.tensors = new LinkedHashMap<>(tensors);
            for (Map.Entry<String, CheckpointTensor> entry : this.tensors.entrySet()) {
                if (entry.getKey() == null || entry.getKey().isBlank() || entry.getValue() == null) {
                    throw new IllegalArgumentException("tensor map must contain named, non-null tensors");
                }
            }
        }

        private float[][] matrix(String name, int rows, int columns) {
            CheckpointTensor tensor = require(name, rows, columns);
            float[] values = tensor.values;
            float[][] matrix = new float[rows][columns];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(values, row * columns, matrix[row], 0, columns);
            }
            return matrix;
        }

        private float[] vector(String name, int length) {
            CheckpointTensor tensor = require(name, length);
            return tensor.values.clone();
        }

        private CheckpointTensor require(String name, int... expectedShape) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("architecture returned an empty checkpoint tensor name");
            }
            CheckpointTensor tensor = tensors.get(name);
            if (tensor == null) throw new IllegalArgumentException("missing required Gemma 3 tensor: " + name);
            int[] actual = tensor.shape;
            if (!java.util.Arrays.equals(actual, expectedShape)) {
                throw new IllegalArgumentException("tensor '" + name + "' has shape "
                        + java.util.Arrays.toString(actual) + ", expected "
                        + java.util.Arrays.toString(expectedShape));
            }
            consumed.add(name);
            return tensor;
        }

        private void rejectUnconsumedModelWeights() {
            List<String> unknown = tensors.keySet().stream()
                    .filter(name -> name.startsWith("model.") || name.startsWith("lm_head."))
                    .filter(name -> !consumed.contains(name))
                    .sorted()
                    .toList();
            if (!unknown.isEmpty()) {
                throw new IllegalArgumentException("unmapped Gemma 3 model tensors: " + String.join(", ", unknown));
            }
        }
    }
}
