package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Validated architecture configuration for the text path of EmbeddingGemma 2. */
public record EmbeddingGemma2Config(
        String modelType,
        String textModelType,
        int hiddenSize,
        int intermediateSize,
        int numHiddenLayers,
        int numAttentionHeads,
        int numKeyValueHeads,
        int headDim,
        int hiddenSizePerLayerInput,
        String hiddenActivation,
        int embeddingDim,
        int vocabSize,
        int maxPositionEmbeddings,
        int maxInputTokens,
        String dtype,
        double rmsNormEps,
        Integer slidingWindow,
        List<String> layerTypes,
        Map<String, JsonNode> perLayerConfig,
        JsonNode ropeParameters) {

    public static final String MODEL_TYPE = "embedding_gemma2";
    public static final String TEXT_MODEL_TYPE = "embedding_gemma2_text";
    public static final int DOCUMENTED_MAX_INPUT_TOKENS = 8192;
    public static final int DEFAULT_HIDDEN_SIZE_PER_LAYER_INPUT = 512;
    public static final String DEFAULT_HIDDEN_ACTIVATION = "gelu_pytorch_tanh";

    public EmbeddingGemma2Config {
        modelType = requireText(modelType, "model_type");
        textModelType = requireText(textModelType, "text_config.model_type");
        dtype = requireText(dtype, "text_config.dtype");
        hiddenActivation = requireText(hiddenActivation, "text_config.hidden_activation");
        if (hiddenSizePerLayerInput <= 0) {
            throw new IllegalArgumentException("hidden_size_per_layer_input must be positive");
        }
        layerTypes = layerTypes == null ? List.of() : List.copyOf(layerTypes);
        perLayerConfig = immutableJsonMap(perLayerConfig);
        ropeParameters = ropeParameters == null ? null : ropeParameters.deepCopy();
    }

    /** Parses nested text settings without discarding per-layer or RoPE overrides. */
    public static EmbeddingGemma2Config fromJson(JsonNode root) {
        Objects.requireNonNull(root, "root config must not be null");
        JsonNode text = root.path("text_config");
        if (!text.isObject()) {
            throw new IllegalArgumentException("EmbeddingGemma 2 requires an object text_config");
        }

        String modelType = textValue(root, "model_type");
        String textModelType = textValue(text, "model_type");
        requireEquals(MODEL_TYPE, modelType, "model_type");
        requireEquals(TEXT_MODEL_TYPE, textModelType, "text_config.model_type");

        int layers = positiveInt(text, "num_hidden_layers");
        int hidden = positiveInt(text, "hidden_size");
        int intermediate = positiveInt(text, "intermediate_size");
        int queryHeads = positiveInt(text, "num_attention_heads");
        int kvHeads = positiveInt(text, "num_key_value_heads");
        int headDim = positiveInt(text, "head_dim");
        int hiddenPerLayer = optionalPositiveInt(text, "hidden_size_per_layer_input",
                DEFAULT_HIDDEN_SIZE_PER_LAYER_INPUT);
        String hiddenActivation = optionalText(text, "hidden_activation", DEFAULT_HIDDEN_ACTIVATION);
        if (!DEFAULT_HIDDEN_ACTIVATION.equals(hiddenActivation)) {
            throw new IllegalArgumentException("unsupported EmbeddingGemma 2 hidden_activation: " + hiddenActivation);
        }
        int embeddingDim = positiveInt(text, "embedding_dim");
        int vocab = positiveInt(text, "vocab_size");
        int maxPositions = positiveInt(text, "max_position_embeddings");
        double normEps = positiveDouble(text, "rms_norm_eps");

        if (queryHeads % kvHeads != 0) {
            throw new IllegalArgumentException(
                    "num_attention_heads must be divisible by num_key_value_heads");
        }

        String dtype = optionalText(text, "dtype", optionalText(text, "torch_dtype", null));
        if (!"bfloat16".equalsIgnoreCase(dtype) && !"float32".equalsIgnoreCase(dtype)) {
            throw new IllegalArgumentException(
                    "EmbeddingGemma 2 text path supports bfloat16 or float32, not: " + dtype);
        }

        Integer slidingWindow = optionalPositiveInt(text, "sliding_window");
        List<String> layerTypes = stringList(text.path("layer_types"), "text_config.layer_types");
        if (!layerTypes.isEmpty() && layerTypes.size() != layers) {
            throw new IllegalArgumentException("layer_types has " + layerTypes.size()
                    + " entries but num_hidden_layers is " + layers);
        }

        Map<String, JsonNode> overrides = jsonObjectMap(text.path("per_layer_config"));
        for (String key : overrides.keySet()) {
            int layerIndex;
            try {
                layerIndex = Integer.parseInt(key);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("per_layer_config key must be a layer index: " + key, e);
            }
            if (layerIndex < 0 || layerIndex >= layers) {
                throw new IllegalArgumentException("per_layer_config layer index out of range: " + key);
            }
        }

        JsonNode ropeParameters = text.path("rope_parameters");
        if (!ropeParameters.isObject()
                || !ropeParameters.path("full_attention").isObject()
                || !ropeParameters.path("sliding_attention").isObject()) {
            throw new IllegalArgumentException(
                    "text_config.rope_parameters must define full_attention and sliding_attention");
        }
        positiveDouble(ropeParameters.path("full_attention"), "rope_theta");
        positiveDouble(ropeParameters.path("sliding_attention"), "rope_theta");

        return new EmbeddingGemma2Config(modelType, textModelType, hidden, intermediate, layers,
                queryHeads, kvHeads, headDim, hiddenPerLayer, hiddenActivation, embeddingDim, vocab,
                maxPositions, DOCUMENTED_MAX_INPUT_TOKENS, dtype, normEps, slidingWindow, layerTypes,
                normalizeLayerKeys(overrides), ropeParameters);
    }

    public Map<String, JsonNode> perLayerConfig() {
        return immutableJsonMap(perLayerConfig);
    }

    public JsonNode ropeParameters() {
        return ropeParameters == null ? null : ropeParameters.deepCopy();
    }

    /** Returns a defensive copy of one layer's raw override, or null. */
    public JsonNode perLayerOverride(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= numHiddenLayers) {
            throw new IndexOutOfBoundsException("layer index out of range: " + layerIndex);
        }
        JsonNode override = perLayerConfig.get(Integer.toString(layerIndex));
        return override == null ? null : override.deepCopy();
    }

    private static Map<String, JsonNode> normalizeLayerKeys(Map<String, JsonNode> source) {
        Map<String, JsonNode> normalized = new LinkedHashMap<>();
        source.forEach((key, value) -> normalized.put(Integer.toString(Integer.parseInt(key)), value));
        return normalized;
    }

    private static Map<String, JsonNode> immutableJsonMap(Map<String, JsonNode> source) {
        if (source == null || source.isEmpty()) return Map.of();
        Map<String, JsonNode> copy = new LinkedHashMap<>();
        source.forEach((key, value) -> copy.put(key, value == null ? null : value.deepCopy()));
        return Collections.unmodifiableMap(copy);
    }

    private static Map<String, JsonNode> jsonObjectMap(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return Map.of();
        if (!node.isObject()) {
            throw new IllegalArgumentException("text_config.per_layer_config must be an object");
        }
        Map<String, JsonNode> result = new LinkedHashMap<>();
        node.fields().forEachRemaining(entry -> result.put(entry.getKey(), entry.getValue().deepCopy()));
        return result;
    }

    private static List<String> stringList(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) return List.of();
        if (!node.isArray()) throw new IllegalArgumentException(field + " must be an array");
        List<String> values = new ArrayList<>();
        node.forEach(value -> {
            if (!value.isTextual() || value.asText().isBlank()) {
                throw new IllegalArgumentException(field + " must contain non-blank strings");
            }
            values.add(value.asText());
        });
        return List.copyOf(values);
    }

    private static int positiveInt(JsonNode node, String field) {
        JsonNode value = node.path(field);
        if (!value.canConvertToInt() || value.asInt() <= 0) {
            throw new IllegalArgumentException(field + " must be a positive integer");
        }
        return value.asInt();
    }

    private static Integer optionalPositiveInt(JsonNode node, String field) {
        return optionalPositiveInt(node, field, null);
    }

    private static int optionalPositiveInt(JsonNode node, String field, int fallback) {
        Integer value = optionalPositiveInt(node, field);
        return value == null ? fallback : value;
    }

    private static Integer optionalPositiveInt(JsonNode node, String field, Integer fallback) {
        JsonNode value = node.path(field);
        if (value.isMissingNode() || value.isNull()) return fallback;
        if (!value.canConvertToInt() || value.asInt() <= 0) {
            throw new IllegalArgumentException(field + " must be a positive integer when present");
        }
        return value.asInt();
    }

    private static double positiveDouble(JsonNode node, String field) {
        JsonNode value = node.path(field);
        if (!value.isNumber() || !Double.isFinite(value.asDouble()) || value.asDouble() <= 0.0d) {
            throw new IllegalArgumentException(field + " must be a finite positive number");
        }
        return value.asDouble();
    }

    private static String optionalText(JsonNode node, String field, String fallback) {
        JsonNode value = node.path(field);
        return value.isTextual() && !value.asText().isBlank() ? value.asText() : fallback;
    }

    private static String textValue(JsonNode node, String field) {
        String value = optionalText(node, field, null);
        if (value == null) throw new IllegalArgumentException(field + " must be a non-blank string");
        return value;
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(field + " must not be blank");
        return value;
    }

    private static void requireEquals(String expected, String actual, String field) {
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(field + " must be '" + expected + "' but was '" + actual + "'");
        }
    }
}
