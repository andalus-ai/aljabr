package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Objects;

/**
 * Resolves the checkpoint's mixed attention layout into immutable, per-layer
 * execution metadata. This class does not create tensors or imply runner readiness.
 */
public final class EmbeddingGemma2LayerConfigResolver {

    private EmbeddingGemma2LayerConfigResolver() {
    }

    public static EmbeddingGemma2LayerConfig resolve(EmbeddingGemma2Config config, int layerIndex) {
        Objects.requireNonNull(config, "config must not be null");
        if (layerIndex < 0 || layerIndex >= config.numHiddenLayers()) {
            throw new IndexOutOfBoundsException("layer index out of range: " + layerIndex);
        }

        List<String> layerTypes = config.layerTypes();
        if (layerTypes.size() != config.numHiddenLayers()) {
            throw new IllegalStateException(
                    "layer_types must be present and complete before resolving attention layers");
        }

        EmbeddingGemma2LayerConfig.AttentionType attentionType = switch (layerTypes.get(layerIndex)) {
            case "full_attention" -> EmbeddingGemma2LayerConfig.AttentionType.FULL;
            case "sliding_attention" -> EmbeddingGemma2LayerConfig.AttentionType.SLIDING;
            default -> throw new IllegalArgumentException(
                    "unsupported layer type at index " + layerIndex + ": " + layerTypes.get(layerIndex));
        };

        JsonNode override = config.perLayerOverride(layerIndex);
        int headDim = positiveOverride(override, "head_dim", config.headDim());
        int kvHeads = positiveOverride(override, "num_key_value_heads", config.numKeyValueHeads());
        if (config.numAttentionHeads() % kvHeads != 0) {
            throw new IllegalArgumentException(
                    "effective num_attention_heads must be divisible by per-layer num_key_value_heads"
                            + " at layer " + layerIndex);
        }

        JsonNode rope = config.ropeParameters().path(
                attentionType == EmbeddingGemma2LayerConfig.AttentionType.FULL
                        ? "full_attention" : "sliding_attention");
        double ropeTheta = positiveNumber(rope, "rope_theta");
        String ropeType = requiredText(rope, "rope_type");
        Double partialRotaryFactor = optionalNumber(rope, "partial_rotary_factor");
        Integer slidingWindow = attentionType == EmbeddingGemma2LayerConfig.AttentionType.SLIDING
                ? config.slidingWindow() : null;
        if (attentionType == EmbeddingGemma2LayerConfig.AttentionType.SLIDING && slidingWindow == null) {
            throw new IllegalStateException("sliding_window is required for sliding attention layers");
        }

        return new EmbeddingGemma2LayerConfig(layerIndex, attentionType,
                config.numAttentionHeads(), kvHeads, headDim, ropeTheta, ropeType,
                partialRotaryFactor, slidingWindow);
    }

    private static int positiveOverride(JsonNode override, String field, int fallback) {
        if (override == null || override.isMissingNode() || override.isNull()
                || !override.has(field)) {
            return fallback;
        }
        JsonNode value = override.path(field);
        if (!value.canConvertToInt() || value.asInt() <= 0) {
            throw new IllegalArgumentException("per-layer " + field + " must be a positive integer");
        }
        return value.asInt();
    }

    private static double positiveNumber(JsonNode node, String field) {
        JsonNode value = node.path(field);
        if (!value.isNumber() || !Double.isFinite(value.asDouble()) || value.asDouble() <= 0.0d) {
            throw new IllegalArgumentException(field + " must be a finite positive number");
        }
        return value.asDouble();
    }

    private static Double optionalNumber(JsonNode node, String field) {
        JsonNode value = node.path(field);
        if (value.isMissingNode() || value.isNull()) return null;
        if (!value.isNumber() || !Double.isFinite(value.asDouble())
                || value.asDouble() <= 0.0d || value.asDouble() > 1.0d) {
            throw new IllegalArgumentException(field + " must be in (0, 1]");
        }
        return value.asDouble();
    }

    private static String requiredText(JsonNode node, String field) {
        JsonNode value = node.path(field);
        if (!value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException(field + " must be a non-blank string");
        }
        return value.asText();
    }
}
