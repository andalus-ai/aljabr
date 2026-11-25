package tech.kayys.aljabr.models.chronos2;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Describes the exact Hugging Face tensor names and shapes expected by the
 * Chronos-2 architecture, without coupling the model family to a particular
 * Safetensors reader or tensor storage implementation.
 *
 * <p>The upstream checkpoint parameter names already match the model's logical
 * parameter paths, so this mapper validates and binds names rather than
 * renaming or copying tensor payloads.
 */
public final class Chronos2CheckpointMapper {

    public CheckpointMapping map(Chronos2Config config, Map<String, long[]> checkpointShapes) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(checkpointShapes, "checkpointShapes");
        Map<String, TensorSpec> expected = expectedTensors(config);
        Map<String, String> bindings = new LinkedHashMap<>();
        Map<String, String> mismatches = new LinkedHashMap<>();

        for (TensorSpec spec : expected.values()) {
            long[] actual = checkpointShapes.get(spec.checkpointName());
            if (actual == null) {
                mismatches.put(spec.checkpointName(), "missing; expected " + format(spec.shape()));
            } else if (!java.util.Arrays.equals(spec.shape(), actual)) {
                mismatches.put(spec.checkpointName(),
                        "expected " + format(spec.shape()) + " but found " + format(actual));
            } else {
                bindings.put(spec.checkpointName(), spec.parameterName());
            }
        }
        if (!mismatches.isEmpty()) {
            throw new IllegalArgumentException("Invalid Chronos-2 checkpoint tensors: " + mismatches);
        }

        return new CheckpointMapping(bindings, expected.size(), checkpointShapes.size());
    }

    public Map<String, TensorSpec> expectedTensors(Chronos2Config config) {
        Objects.requireNonNull(config, "config");
        Map<String, TensorSpec> specs = new LinkedHashMap<>();
        int model = config.modelDimension();
        int inner = Math.multiplyExact(config.headCount(), config.keyValueDimension());
        int patchFeatures = Math.multiplyExact(config.inputPatchSize(), 3);
        int outputFeatures = Math.multiplyExact(config.quantiles().size(), config.outputPatchSize());

        add(specs, "shared.weight", config.useRegressionToken() ? 2 : 1, model);
        addResidualBlock(specs, "input_patch_embedding", patchFeatures, config.feedForwardDimension(), model);
        addResidualBlock(specs, "output_patch_embedding", model, config.feedForwardDimension(), outputFeatures);

        for (int layer = 0; layer < config.layerCount(); layer++) {
            String time = "encoder.block." + layer + ".layer.0";
            String group = "encoder.block." + layer + ".layer.1";
            String feedForward = "encoder.block." + layer + ".layer.2";

            addAttention(specs, time + ".self_attention", model, inner);
            add(specs, time + ".layer_norm.weight", model);

            addAttention(specs, group + ".self_attention", model, inner);
            add(specs, group + ".layer_norm.weight", model);

            add(specs, feedForward + ".layer_norm.weight", model);
            add(specs, feedForward + ".mlp.wi.weight", config.feedForwardDimension(), model);
            add(specs, feedForward + ".mlp.wo.weight", model, config.feedForwardDimension());
        }
        add(specs, "encoder.final_layer_norm.weight", model);
        return Map.copyOf(specs);
    }

    private static void addAttention(Map<String, TensorSpec> specs, String prefix, int model, int inner) {
        add(specs, prefix + ".q.weight", inner, model);
        add(specs, prefix + ".k.weight", inner, model);
        add(specs, prefix + ".v.weight", inner, model);
        add(specs, prefix + ".o.weight", model, inner);
    }

    private static void addResidualBlock(
            Map<String, TensorSpec> specs, String prefix, int input, int hidden, int output) {
        add(specs, prefix + ".hidden_layer.weight", hidden, input);
        add(specs, prefix + ".hidden_layer.bias", hidden);
        add(specs, prefix + ".output_layer.weight", output, hidden);
        add(specs, prefix + ".output_layer.bias", output);
        add(specs, prefix + ".residual_layer.weight", output, input);
        add(specs, prefix + ".residual_layer.bias", output);
    }

    private static void add(Map<String, TensorSpec> specs, String name, long... shape) {
        specs.put(name, new TensorSpec(name, name, shape));
    }

    private static String format(long[] shape) {
        return java.util.Arrays.toString(shape);
    }

    /** A single source tensor and its destination logical parameter. */
    public record TensorSpec(String checkpointName, String parameterName, long[] shape) {
        public TensorSpec {
            Objects.requireNonNull(checkpointName, "checkpointName");
            Objects.requireNonNull(parameterName, "parameterName");
            shape = Objects.requireNonNull(shape, "shape").clone();
            if (checkpointName.isBlank() || parameterName.isBlank() || shape.length == 0) {
                throw new IllegalArgumentException("tensor names and shapes must be non-empty");
            }
            for (long dimension : shape) {
                if (dimension < 1) {
                    throw new IllegalArgumentException("tensor dimensions must be positive");
                }
            }
        }

        @Override public long[] shape() { return shape.clone(); }
    }

    /** Validated zero-copy name bindings; tensor payload ownership stays with the loader. */
    public record CheckpointMapping(
            Map<String, String> parameterBindings,
            int expectedTensorCount,
            int checkpointTensorCount) {
        public CheckpointMapping {
            parameterBindings = Map.copyOf(Objects.requireNonNull(parameterBindings, "parameterBindings"));
            if (expectedTensorCount < 1 || parameterBindings.size() != expectedTensorCount
                    || checkpointTensorCount < expectedTensorCount) {
                throw new IllegalArgumentException("checkpoint mapping is incomplete");
            }
        }
    }
}
