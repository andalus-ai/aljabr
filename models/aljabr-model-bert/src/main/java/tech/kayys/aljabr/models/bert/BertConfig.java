package tech.kayys.aljabr.models.bert;

import java.util.Map;
import java.util.Objects;

/** Immutable, validated configuration for BERT encoder-family checkpoints. */
public record BertConfig(
        int vocabularySize,
        int hiddenSize,
        int layerCount,
        int attentionHeadCount,
        int intermediateSize,
        int maxPositionEmbeddings,
        int typeVocabularySize,
        int paddingTokenId,
        double layerNormEpsilon,
        String hiddenActivation) {

    public BertConfig {
        requirePositive("vocabularySize", vocabularySize);
        requirePositive("hiddenSize", hiddenSize);
        requirePositive("layerCount", layerCount);
        requirePositive("attentionHeadCount", attentionHeadCount);
        requirePositive("intermediateSize", intermediateSize);
        requirePositive("maxPositionEmbeddings", maxPositionEmbeddings);
        requirePositive("typeVocabularySize", typeVocabularySize);
        if (hiddenSize % attentionHeadCount != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by attentionHeadCount");
        }
        if (paddingTokenId < 0 || paddingTokenId >= vocabularySize) {
            throw new IllegalArgumentException("paddingTokenId must be within the vocabulary");
        }
        if (!Double.isFinite(layerNormEpsilon) || layerNormEpsilon <= 0.0d) {
            throw new IllegalArgumentException("layerNormEpsilon must be finite and positive");
        }
        hiddenActivation = Objects.requireNonNull(hiddenActivation, "hiddenActivation").trim();
        if (hiddenActivation.isEmpty()) {
            throw new IllegalArgumentException("hiddenActivation must not be blank");
        }
    }

    /** Reads canonical Hugging Face BERT config keys and documented defaults. */
    public static BertConfig from(Map<String, ?> values) {
        Objects.requireNonNull(values, "values");
        return new BertConfig(
                requiredInt(values, "vocab_size"),
                requiredInt(values, "hidden_size"),
                requiredInt(values, "num_hidden_layers"),
                requiredInt(values, "num_attention_heads"),
                requiredInt(values, "intermediate_size"),
                intValue(values, "max_position_embeddings", 512),
                intValue(values, "type_vocab_size", 2),
                intValue(values, "pad_token_id", 0),
                doubleValue(values, "layer_norm_eps", 1.0e-12d),
                stringValue(values, "hidden_act", "gelu"));
    }

    public int headDimension() {
        return hiddenSize / attentionHeadCount;
    }

    private static void requirePositive(String name, int value) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
    }

    private static int requiredInt(Map<String, ?> values, String key) {
        if (!values.containsKey(key) || values.get(key) == null) {
            throw new IllegalArgumentException("Missing required BERT config field: " + key);
        }
        return intValue(values, key, 0);
    }

    private static int intValue(Map<String, ?> values, String key, int defaultValue) {
        Object value = values.get(key);
        if (value == null) return defaultValue;
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException("BERT config field " + key + " must be numeric");
        }
        double numeric = number.doubleValue();
        if (!Double.isFinite(numeric) || numeric != Math.rint(numeric)
                || numeric < Integer.MIN_VALUE || numeric > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("BERT config field " + key + " must be an integer");
        }
        return number.intValue();
    }

    private static double doubleValue(Map<String, ?> values, String key, double defaultValue) {
        Object value = values.get(key);
        if (value == null) return defaultValue;
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException("BERT config field " + key + " must be numeric");
        }
        return number.doubleValue();
    }

    private static String stringValue(Map<String, ?> values, String key, String defaultValue) {
        Object value = values.get(key);
        if (value == null) return defaultValue;
        if (!(value instanceof String string)) {
            throw new IllegalArgumentException("BERT config field " + key + " must be a string");
        }
        return string;
    }
}
