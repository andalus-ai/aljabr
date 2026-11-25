package tech.kayys.aljabr.models.bert;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Storage-independent mapper for Hugging Face BERT encoder checkpoints.
 *
 * <p>Maps canonical Hugging Face tensor names to Aljabr's reusable BERT
 * module state names. Dense weights retain their native [out, in] layout.
 * Optional pooler and position/token-type ID buffers are ignored because the
 * sentence-embedding encoder does not consume them.</p>
 */
public final class BertCheckpointMapper {
    private static final String[] IGNORED_PREFIXES = {"pooler.", "cls."};
    private static final java.util.Set<String> IGNORED_BUFFERS = java.util.Set.of(
            "embeddings.position_ids", "embeddings.token_type_ids");

    public Map<String, CheckpointTensor> map(BertConfig config,
            Map<String, CheckpointTensor> checkpointTensors) {
        Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(checkpointTensors, "checkpointTensors must not be null");
        Map<String, int[]> expected = expectedTensorShapes(config);
        Map<String, CheckpointTensor> mapped = new LinkedHashMap<>();
        for (Map.Entry<String, String> binding : parameterBindings(config).entrySet()) {
            CheckpointTensor tensor = checkpointTensors.get(binding.getKey());
            if (tensor == null) {
                throw new IllegalArgumentException("Missing required BERT checkpoint tensor: " + binding.getKey());
            }
            int[] expectedShape = expected.get(binding.getKey());
            if (!java.util.Arrays.equals(expectedShape, tensor.shape())) {
                throw new IllegalArgumentException(binding.getKey() + " expected shape "
                        + java.util.Arrays.toString(expectedShape) + " but was "
                        + java.util.Arrays.toString(tensor.shape()));
            }
            mapped.put(binding.getValue(), tensor);
        }
        for (String name : checkpointTensors.keySet()) {
            if (parameterBindings(config).containsKey(name) || IGNORED_BUFFERS.contains(name)
                    || isIgnoredHead(name)) continue;
            throw new IllegalArgumentException("Unsupported or unmapped BERT checkpoint tensor: " + name);
        }
        return Map.copyOf(mapped);
    }

    /** Expected source tensor shapes keyed by canonical Hugging Face names. */
    public Map<String, int[]> expectedTensorShapes(BertConfig config) {
        Objects.requireNonNull(config, "config");
        Map<String, int[]> shapes = new LinkedHashMap<>();
        add(shapes, "embeddings.word_embeddings.weight", config.vocabularySize(), config.hiddenSize());
        add(shapes, "embeddings.position_embeddings.weight",
                config.maxPositionEmbeddings(), config.hiddenSize());
        add(shapes, "embeddings.token_type_embeddings.weight",
                config.typeVocabularySize(), config.hiddenSize());
        add(shapes, "embeddings.LayerNorm.weight", config.hiddenSize());
        add(shapes, "embeddings.LayerNorm.bias", config.hiddenSize());
        for (int layer = 0; layer < config.layerCount(); layer++) {
            String source = "encoder.layer." + layer + ".";
            addLinear(shapes, source + "attention.self.query.", config.hiddenSize(), config.hiddenSize());
            addLinear(shapes, source + "attention.self.key.", config.hiddenSize(), config.hiddenSize());
            addLinear(shapes, source + "attention.self.value.", config.hiddenSize(), config.hiddenSize());
            addLinear(shapes, source + "attention.output.dense.", config.hiddenSize(), config.hiddenSize());
            addNorm(shapes, source + "attention.output.LayerNorm.", config.hiddenSize());
            addLinear(shapes, source + "intermediate.dense.", config.intermediateSize(), config.hiddenSize());
            addLinear(shapes, source + "output.dense.", config.hiddenSize(), config.intermediateSize());
            addNorm(shapes, source + "output.LayerNorm.", config.hiddenSize());
        }
        Map<String, int[]> defensive = new LinkedHashMap<>();
        shapes.forEach((name, shape) -> defensive.put(name, shape.clone()));
        return java.util.Collections.unmodifiableMap(defensive);
    }

    /** Source Hugging Face names to destination NNModule state names. */
    public Map<String, String> parameterBindings(BertConfig config) {
        Objects.requireNonNull(config, "config");
        Map<String, String> names = new LinkedHashMap<>();
        bind(names, "embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight");
        bind(names, "embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight");
        bind(names, "embeddings.token_type_embeddings.weight", "embeddings.token_type_embeddings.weight");
        bind(names, "embeddings.LayerNorm.weight", "embeddings.layer_norm.weight");
        bind(names, "embeddings.LayerNorm.bias", "embeddings.layer_norm.bias");
        for (int layer = 0; layer < config.layerCount(); layer++) {
            String source = "encoder.layer." + layer + ".";
            String target = "encoder_layer_" + layer + ".";
            bindLinear(names, source + "attention.self.query.", target + "self_attn.query.");
            bindLinear(names, source + "attention.self.key.", target + "self_attn.key.");
            bindLinear(names, source + "attention.self.value.", target + "self_attn.value.");
            bindLinear(names, source + "attention.output.dense.", target + "self_attn.output.");
            bindNorm(names, source + "attention.output.LayerNorm.", target + "norm1.");
            bindLinear(names, source + "intermediate.dense.", target + "linear1.");
            bindLinear(names, source + "output.dense.", target + "linear2.");
            bindNorm(names, source + "output.LayerNorm.", target + "norm2.");
        }
        return java.util.Collections.unmodifiableMap(names);
    }

    private static void addLinear(Map<String, int[]> shapes, String prefix, int out, int in) {
        add(shapes, prefix + "weight", out, in);
        add(shapes, prefix + "bias", out);
    }

    private static void addNorm(Map<String, int[]> shapes, String prefix, int size) {
        add(shapes, prefix + "weight", size);
        add(shapes, prefix + "bias", size);
    }

    private static void add(Map<String, int[]> shapes, String name, int... dimensions) {
        shapes.put(name, dimensions);
    }

    private static void bindLinear(Map<String, String> names, String source, String target) {
        bind(names, source + "weight", target + "weight");
        bind(names, source + "bias", target + "bias");
    }

    private static void bindNorm(Map<String, String> names, String source, String target) {
        bind(names, source + "weight", target + "weight");
        bind(names, source + "bias", target + "bias");
    }

    private static void bind(Map<String, String> names, String source, String target) {
        names.put(source, target);
    }

    private static boolean isIgnoredHead(String name) {
        for (String prefix : IGNORED_PREFIXES) if (name.startsWith(prefix)) return true;
        return false;
    }

    /** Immutable dense F32 tensor payload independent of the checkpoint format. */
    public static final class CheckpointTensor {
        private final int[] shape;
        private final float[] values;

        public CheckpointTensor(int[] shape, float[] values) {
            this.shape = Objects.requireNonNull(shape, "shape").clone();
            this.values = Objects.requireNonNull(values, "values").clone();
            long count = 1;
            for (int dimension : this.shape) {
                if (dimension <= 0) throw new IllegalArgumentException("checkpoint dimensions must be positive");
                count = Math.multiplyExact(count, dimension);
            }
            if (count != this.values.length) {
                throw new IllegalArgumentException("checkpoint tensor shape does not match value count");
            }
            for (float value : this.values) {
                if (!Float.isFinite(value)) throw new IllegalArgumentException("checkpoint values must be finite");
            }
        }

        public int[] shape() { return shape.clone(); }
        public float[] values() { return values.clone(); }
    }
}
