package tech.kayys.aljabr.gguf.api;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Immutable, metadata-only description of a GGUF artifact.
 *
 * <p>This descriptor records observable artifact facts. It does not infer
 * runtime capabilities from an architecture, model name or tensor inventory.</p>
 */
public record GgufModelDescriptor(
        String architecture,
        String modelName,
        Set<String> metadataKeys,
        Set<String> tensorNames,
        boolean tokenizerMetadataPresent,
        boolean chatTemplatePresent,
        Map<String, GgufTensorDescriptor> tensorDescriptors) {

    private static final String ARCHITECTURE_KEY = "general.architecture";
    private static final String MODEL_NAME_KEY = "general.name";
    private static final Set<String> TOKENIZER_KEYS = Set.of(
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.pre");

    /** Preserves the original constructor for callers that only have tensor names. */
    public GgufModelDescriptor(
            String architecture,
            String modelName,
            Set<String> metadataKeys,
            Set<String> tensorNames,
            boolean tokenizerMetadataPresent,
            boolean chatTemplatePresent) {
        this(architecture, modelName, metadataKeys, tensorNames,
                tokenizerMetadataPresent, chatTemplatePresent, Map.of());
    }

    public GgufModelDescriptor {
        architecture = normalize(architecture);
        modelName = normalize(modelName);
        metadataKeys = immutableSorted(metadataKeys, "metadataKeys");
        tensorNames = immutableSorted(tensorNames, "tensorNames");
        Objects.requireNonNull(tensorDescriptors, "tensorDescriptors");
        TreeMap<String, GgufTensorDescriptor> descriptors = new TreeMap<>();
        tensorDescriptors.forEach((name, descriptor) -> {
            Objects.requireNonNull(name, "tensor descriptor name");
            Objects.requireNonNull(descriptor, "tensor descriptor");
            if (!name.equals(descriptor.name())) {
                throw new IllegalArgumentException("Tensor descriptor key does not match its name: " + name);
            }
            if (!tensorNames.contains(name)) {
                throw new IllegalArgumentException("Tensor descriptor is absent from tensorNames: " + name);
            }
            descriptors.put(name, descriptor);
        });
        tensorDescriptors = Collections.unmodifiableMap(descriptors);
    }

    /** Builds a descriptor from parsed metadata and tensor names only. */
    public static GgufModelDescriptor inspect(
            Map<String, ?> metadata, Collection<String> tensorNames) {
        Objects.requireNonNull(tensorNames, "tensorNames");
        Set<String> names = new TreeSet<>();
        for (String tensorName : tensorNames) {
            names.add(Objects.requireNonNull(tensorName, "tensor name"));
        }
        return create(metadata, names, Map.of());
    }

    /** Builds a descriptor retaining dimensions and GGML type ids from the parser. */
    public static GgufModelDescriptor inspectWithTensors(
            Map<String, ?> metadata, Collection<GgufTensorDescriptor> tensors) {
        Objects.requireNonNull(tensors, "tensors");
        Map<String, GgufTensorDescriptor> descriptors = new TreeMap<>();
        for (GgufTensorDescriptor tensor : tensors) {
            Objects.requireNonNull(tensor, "tensor");
            if (descriptors.putIfAbsent(tensor.name(), tensor) != null) {
                throw new IllegalArgumentException("Duplicate tensor descriptor: " + tensor.name());
            }
        }
        return create(metadata, descriptors.keySet(), descriptors);
    }

    private static GgufModelDescriptor create(
            Map<String, ?> metadata,
            Set<String> tensorNames,
            Map<String, GgufTensorDescriptor> tensorDescriptors) {
        Objects.requireNonNull(metadata, "metadata");
        String architecture = stringValue(metadata.get(ARCHITECTURE_KEY));
        String modelName = stringValue(metadata.get(MODEL_NAME_KEY));
        Set<String> keys = new TreeSet<>();
        for (String key : metadata.keySet()) {
            keys.add(Objects.requireNonNull(key, "metadata key"));
        }
        boolean tokenizerPresent = TOKENIZER_KEYS.stream().anyMatch(keys::contains);
        boolean chatTemplatePresent = hasNonBlankString(metadata.get("tokenizer.chat_template"));
        return new GgufModelDescriptor(
                architecture, modelName, keys, tensorNames,
                tokenizerPresent, chatTemplatePresent, tensorDescriptors);
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
                + ", tensorDetails=" + tensorDescriptors.size()
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
        if (value == null) return null;
        String normalized = value.trim();
        return normalized.isEmpty() ? null : normalized;
    }

    private static String display(String value) {
        return value == null ? "unknown" : value;
    }
}
