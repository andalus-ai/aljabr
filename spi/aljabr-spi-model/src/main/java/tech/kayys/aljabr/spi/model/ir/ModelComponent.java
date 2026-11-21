package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A model-level tokenizer, modality processor, or other reusable component.
 * Component configuration is typed, while assets and graph bindings are
 * references resolved by loaders or execution planners.
 */
public record ModelComponent(
        String id,
        ModelComponentKind kind,
        Map<String, ConfigurationValue> configuration,
        List<String> assetIds,
        List<ComponentGraphBinding> graphBindings,
        Map<String, String> extensions) {
    public ModelComponent {
        id = requireText(id, "id");
        Objects.requireNonNull(kind, "kind must not be null");
        configuration = immutableConfiguration(configuration);
        assetIds = immutableTextList(assetIds, "assetIds");
        graphBindings = List.copyOf(Objects.requireNonNull(graphBindings, "graphBindings must not be null"));
        if (graphBindings.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("graphBindings must not contain null");
        }
        extensions = immutableStringMap(extensions, "extensions");
    }

    public static ModelComponent of(String id, ModelComponentKind kind) {
        return new ModelComponent(id, kind, Map.of(), List.of(), List.of(), Map.of());
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }

    private static List<String> immutableTextList(List<String> values, String name) {
        List<String> copy = List.copyOf(Objects.requireNonNull(values, name + " must not be null"));
        if (copy.stream().anyMatch(value -> value.isBlank())) {
            throw new IllegalArgumentException(name + " must not contain blank values");
        }
        return copy;
    }

    private static Map<String, ConfigurationValue> immutableConfiguration(
            Map<String, ConfigurationValue> values) {
        Objects.requireNonNull(values, "configuration must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank()
                || e.getValue() == null)) {
            throw new IllegalArgumentException("configuration must contain non-blank keys and non-null values");
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }

    private static Map<String, String> immutableStringMap(Map<String, String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank()
                || e.getValue() == null)) {
            throw new IllegalArgumentException(name + " must contain non-blank keys and non-null values");
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }
}
