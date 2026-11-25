package tech.kayys.aljabr.gguf.api;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Immutable, metadata-only description of a GGUF artifact.
 *
 * <p>This descriptor deliberately records observable artifact facts rather
 * than guessing runtime capabilities from an architecture or model name.
 * A model-family adapter may use these facts together with verified tensor
 * mappings and runtime support to make a capability declaration.</p>
 */
public record GgufModelDescriptor(
        String architecture,
        String modelName,
        Set<String> metadataKeys,
        Set<String> tensorNames,
        boolean tokenizerMetadataPresent,
        boolean chatTemplatePresent) {

    private static final String ARCHITECTURE_KEY = "general.architecture";
    private static final String MODEL_NAME_KEY = "general.name";
    private static final Set<String> TOKENIZER_KEYS = Set.of(
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.pre");

    public GgufModelDescriptor {
        architecture = normalize(architecture);
        modelName = normalize(modelName);
        metadataKeys = immutableSorted(metadataKeys, "metadataKeys");
        tensorNames = immutableSorted(tensorNames, "tensorNames");
    }

    /**
     * Builds a descriptor from parsed GGUF metadata and tensor names.
     * Metadata values are only interpreted for the two descriptive string
     * fields and the presence of tokenizer/chat-template keys.
     */
    public static GgufModelDescriptor inspect(
            Map<String, ?> metadata, Collection<String> tensorNames) {
        Objects.requireNonNull(metadata, "metadata");
        Objects.requireNonNull(tensorNames, "tensorNames");

        String architecture = stringValue(metadata.get(ARCHITECTURE_KEY));
        String modelName = stringValue(metadata.get(MODEL_NAME_KEY));
        Set<String> keys = new TreeSet<>();
        for (String key : metadata.keySet()) {
            keys.add(Objects.requireNonNull(key, "metadata key"));
        }
        Set<String> tensors = new TreeSet<>();
        for (String tensorName : tensorNames) {
            tensors.add(Objects.requireNonNull(tensorName, "tensor name"));
        }

        boolean tokenizerPresent = TOKENIZER_KEYS.stream().anyMatch(keys::contains);
        boolean chatTemplatePresent = hasNonBlankString(metadata.get("tokenizer.chat_template"));
        return new GgufModelDescriptor(
                architecture, modelName, keys, tensors, tokenizerPresent, chatTemplatePresent);
    }

    /** Returns whether any tensor name begins with the supplied prefix. */
    public boolean hasTensorPrefix(String prefix) {
        Objects.requireNonNull(prefix, "prefix");
        return !prefix.isEmpty() && tensorNames.stream().anyMatch(name -> name.startsWith(prefix));
    }

    /** Human-readable facts only; does not claim that inference is supported. */
    public String summary() {
        return "architecture=" + display(architecture)
                + ", modelName=" + display(modelName)
                + ", metadataKeys=" + metadataKeys.size()
                + ", tensors=" + tensorNames.size()
                + ", tokenizerMetadata=" + tokenizerMetadataPresent
                + ", chatTemplate=" + chatTemplatePresent;
    }

    private static Set<String> immutableSorted(Set<String> values, String name) {
        Objects.requireNonNull(values, name);
        TreeSet<String> copy = new TreeSet<>();
        for (String value : values) {
            copy.add(Objects.requireNonNull(value, name + " element"));
        }
        return Collections.unmodifiableSet(copy);
    }

    private static String stringValue(Object value) {
        return value instanceof String text ? normalize(text) : null;
    }

    private static boolean hasNonBlankString(Object value) {
        return value instanceof String text && !text.isBlank();
    }

    private static String normalize(String value) {
        if (value == null) {
            return null;
        }
        String normalized = value.trim();
        return normalized.isEmpty() ? null : normalized;
    }

    private static String display(String value) {
        return value == null ? "unknown" : value;
    }
}
