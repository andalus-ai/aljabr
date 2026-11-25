package tech.kayys.aljabr.models.chronos2;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Validated structural configuration for the Chronos-2 checkpoint.
 *
 * <p>This is deliberately separate from T5 text-generation configuration:
 * Chronos-2 reuses T5-sized transformer dimensions but has a forecasting
 * input/output contract, patching scheme, and quantile head of its own.
 */
public record Chronos2Config(
        int contextLength,
        int inputPatchSize,
        int inputPatchStride,
        int outputPatchSize,
        int maxOutputPatches,
        int timeEncodingScale,
        int modelDimension,
        int keyValueDimension,
        int feedForwardDimension,
        int layerCount,
        int headCount,
        double layerNormEpsilon,
        double ropeTheta,
        boolean useRegressionToken,
        boolean useArcsinh,
        List<Double> quantiles) {

    public static final int DEFAULT_CONTEXT_LENGTH = 8192;
    public static final int DEFAULT_MAX_PREDICTION_LENGTH = 1024;

    public Chronos2Config {
        if (contextLength < 1 || contextLength > DEFAULT_CONTEXT_LENGTH) {
            throw new IllegalArgumentException("contextLength must be in [1, 8192]");
        }
        if (inputPatchSize < 1 || inputPatchStride < 1 || outputPatchSize < 1) {
            throw new IllegalArgumentException("patch sizes and stride must be positive");
        }
        if (inputPatchSize != outputPatchSize) {
            throw new IllegalArgumentException("Chronos-2 requires equal input and output patch sizes");
        }
        if (maxOutputPatches < 1 || (long) maxOutputPatches * outputPatchSize > DEFAULT_MAX_PREDICTION_LENGTH) {
            throw new IllegalArgumentException("configured prediction capacity must be in [1, 1024]");
        }
        if (timeEncodingScale < 1 || modelDimension < 1 || keyValueDimension < 1
                || feedForwardDimension < 1 || layerCount < 1 || headCount < 1) {
            throw new IllegalArgumentException("model dimensions and layer counts must be positive");
        }
        if (modelDimension % headCount != 0) {
            throw new IllegalArgumentException("modelDimension must be divisible by headCount");
        }
        if (!(layerNormEpsilon > 0.0) || !(ropeTheta > 0.0)) {
            throw new IllegalArgumentException("normalization epsilon and RoPE theta must be positive");
        }
        quantiles = List.copyOf(Objects.requireNonNull(quantiles, "quantiles"));
        if (quantiles.isEmpty()) {
            throw new IllegalArgumentException("at least one quantile is required");
        }
        double previous = 0.0;
        for (double quantile : quantiles) {
            if (!(quantile > previous && quantile < 1.0)) {
                throw new IllegalArgumentException("quantiles must be strictly increasing in (0, 1)");
            }
            previous = quantile;
        }
    }

    /**
     * Reads the public Hugging Face config.json object without depending on a
     * particular JSON library. The caller owns JSON parsing and supplies maps.
     */
    public static Chronos2Config fromHuggingFaceConfig(Map<String, ?> root) {
        Objects.requireNonNull(root, "root");
        Object architecture = root.get("architectures");
        if (!(architecture instanceof List<?> architectures)
                || !architectures.contains("Chronos2Model")) {
            throw new IllegalArgumentException("checkpoint architecture must contain Chronos2Model");
        }
        if (!"t5".equals(root.get("model_type"))) {
            throw new IllegalArgumentException("Chronos-2 checkpoint model_type must be t5");
        }
        Map<String, ?> forecast = map(root.get("chronos_config"), "chronos_config");
        int inputPatchSize = integer(forecast, "input_patch_size");
        int outputPatchSize = integer(forecast, "output_patch_size");
        List<Double> quantiles = numbers(forecast.get("quantiles"), "chronos_config.quantiles");
        return new Chronos2Config(
                integer(forecast, "context_length"),
                inputPatchSize,
                integer(forecast, "input_patch_stride"),
                outputPatchSize,
                integer(forecast, "max_output_patches"),
                integer(forecast, "time_encoding_scale"),
                integer(root, "d_model"),
                integer(root, "d_kv"),
                integer(root, "d_ff"),
                integer(root, "num_layers"),
                integer(root, "num_heads"),
                decimal(root, "layer_norm_epsilon"),
                decimal(root, "rope_theta"),
                bool(forecast, "use_reg_token"),
                bool(forecast, "use_arcsinh"),
                quantiles);
    }

    public int maximumPredictionLength() {
        return Math.multiplyExact(maxOutputPatches, outputPatchSize);
    }

    private static Map<String, ?> map(Object value, String name) {
        if (!(value instanceof Map<?, ?> raw)) {
            throw new IllegalArgumentException(name + " must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, ?> result = (Map<String, ?>) raw;
        return result;
    }

    private static int integer(Map<String, ?> values, String key) {
        Object value = values.get(key);
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException("missing numeric config field: " + key);
        }
        return number.intValue();
    }

    private static double decimal(Map<String, ?> values, String key) {
        Object value = values.get(key);
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException("missing numeric config field: " + key);
        }
        return number.doubleValue();
    }

    private static boolean bool(Map<String, ?> values, String key) {
        Object value = values.get(key);
        if (!(value instanceof Boolean result)) {
            throw new IllegalArgumentException("missing boolean config field: " + key);
        }
        return result;
    }

    private static List<Double> numbers(Object value, String name) {
        if (!(value instanceof List<?> values)) {
            throw new IllegalArgumentException(name + " must be an array");
        }
        return values.stream().map(item -> {
            if (!(item instanceof Number number)) {
                throw new IllegalArgumentException(name + " must contain only numbers");
            }
            return number.doubleValue();
        }).toList();
    }
}
