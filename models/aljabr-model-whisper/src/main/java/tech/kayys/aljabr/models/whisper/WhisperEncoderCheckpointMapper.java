package tech.kayys.aljabr.models.whisper;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Describes and validates the Hugging Face Whisper encoder parameter layout.
 *
 * <p>This class deliberately contains no Safetensors or tensor-backend
 * dependency. It is the architecture-owned contract that storage adapters
 * can consume without coupling the model module to a checkpoint format.</p>
 */
public final class WhisperEncoderCheckpointMapper {

    /** Logical Aljabr parameter name and required checkpoint tensor shape. */
    public record TensorSpec(String parameterName, long[] shape) {
        public TensorSpec {
            if (parameterName == null || parameterName.isBlank()) {
                throw new IllegalArgumentException("parameterName must not be blank");
            }
            shape = Objects.requireNonNull(shape, "shape").clone();
            for (long dimension : shape) {
                if (dimension <= 0) {
                    throw new IllegalArgumentException("tensor dimensions must be positive");
                }
            }
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }
    }

    /** Metadata-only tensor description supplied by a storage adapter. */
    public record TensorMetadata(String dtype, long[] shape) {
        public TensorMetadata {
            if (dtype == null || dtype.isBlank()) {
                throw new IllegalArgumentException("dtype must not be blank");
            }
            shape = Objects.requireNonNull(shape, "shape").clone();
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }
    }

    /** Immutable validated mapping from checkpoint names to Aljabr parameters. */
    public record CheckpointMapping(Map<String, TensorSpec> parameterBindings) {
        public CheckpointMapping {
            parameterBindings = Map.copyOf(
                    Objects.requireNonNull(parameterBindings, "parameterBindings"));
        }
    }

    /**
     * Returns the exact encoder tensor contract for the supplied configuration.
     * The returned keys are Hugging Face checkpoint names.
     */
    public Map<String, TensorSpec> expectedTensors(WhisperLargeV3Config config) {
        Objects.requireNonNull(config, "config").requireLargeV3Architecture();
        Map<String, TensorSpec> expected = new LinkedHashMap<>();
        int d = config.dModel();
        int m = config.numMelBins();

        add(expected, "model.encoder.conv1.weight", "encoder.conv1.weight", d, m, 3);
        add(expected, "model.encoder.conv1.bias", "encoder.conv1.bias", d);
        add(expected, "model.encoder.conv2.weight", "encoder.conv2.weight", d, d, 3);
        add(expected, "model.encoder.conv2.bias", "encoder.conv2.bias", d);
        add(expected, "model.encoder.embed_positions.weight",
                "encoder.embed_positions.weight", config.maxSourcePositions(), d);

        for (int layer = 0; layer < config.encoderLayers(); layer++) {
            String hf = "model.encoder.layers." + layer + ".";
            String logical = "encoder.layers." + layer + ".";
            addProjection(expected, hf, logical, "self_attn.k_proj", d, d);
            addProjection(expected, hf, logical, "self_attn.v_proj", d, d);
            addProjection(expected, hf, logical, "self_attn.q_proj", d, d);
            addProjection(expected, hf, logical, "self_attn.out_proj", d, d);
            addNorm(expected, hf, logical, "self_attn_layer_norm", d);
            add(expected, hf + "fc1.weight", logical + "fc1.weight", config.encoderFfnDim(), d);
            add(expected, hf + "fc1.bias", logical + "fc1.bias", config.encoderFfnDim());
            add(expected, hf + "fc2.weight", logical + "fc2.weight", d, config.encoderFfnDim());
            add(expected, hf + "fc2.bias", logical + "fc2.bias", d);
            addNorm(expected, hf, logical, "final_layer_norm", d);
        }

        addNorm(expected, "model.encoder.", "encoder.", "layer_norm", d);
        return Map.copyOf(expected);
    }

    /**
     * Validates names, supported floating-point dtypes, and exact dimensions.
     * Extra checkpoint tensors are permitted because a full HF checkpoint also
     * contains the decoder and output projection.
     */
    public CheckpointMapping mapMetadata(
            WhisperLargeV3Config config, Map<String, TensorMetadata> checkpoint) {
        Objects.requireNonNull(checkpoint, "checkpoint");
        Map<String, TensorSpec> expected = expectedTensors(config);
        Map<String, TensorSpec> bindings = new LinkedHashMap<>();
        for (Map.Entry<String, TensorSpec> entry : expected.entrySet()) {
            TensorMetadata actual = checkpoint.get(entry.getKey());
            if (actual == null) {
                throw new IllegalArgumentException("Missing Whisper encoder tensor: " + entry.getKey());
            }
            if (!isSupportedFloatingPoint(actual.dtype())) {
                throw new IllegalArgumentException("Unsupported dtype '" + actual.dtype()
                        + "' for Whisper encoder tensor " + entry.getKey()
                        + "; expected F32, F16, or BF16");
            }
            if (!java.util.Arrays.equals(entry.getValue().shape(), actual.shape())) {
                throw new IllegalArgumentException("Shape mismatch for " + entry.getKey()
                        + ": expected " + java.util.Arrays.toString(entry.getValue().shape())
                        + " but found " + java.util.Arrays.toString(actual.shape()));
            }
            bindings.put(entry.getKey(), entry.getValue());
        }
        return new CheckpointMapping(bindings);
    }

    private static boolean isSupportedFloatingPoint(String dtype) {
        return "F32".equals(dtype) || "F16".equals(dtype) || "BF16".equals(dtype);
    }

    private static void addProjection(Map<String, TensorSpec> target, String hf, String logical,
            String projection, int input, int output) {
        add(target, hf + projection + ".weight", logical + projection + ".weight", output, input);
        add(target, hf + projection + ".bias", logical + projection + ".bias", output);
    }

    private static void addNorm(Map<String, TensorSpec> target, String hf, String logical,
            String norm, int width) {
        add(target, hf + norm + ".weight", logical + norm + ".weight", width);
        add(target, hf + norm + ".bias", logical + norm + ".bias", width);
    }

    private static void add(Map<String, TensorSpec> target, String checkpointName,
            String parameterName, long... shape) {
        if (target.put(checkpointName, new TensorSpec(parameterName, shape)) != null) {
            throw new IllegalStateException("Duplicate Whisper checkpoint tensor: " + checkpointName);
        }
    }
}
