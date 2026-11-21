package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** One operation invocation in a graph; operation identifiers are format-neutral and extensible. */
public record OperationDescriptor(
        String id,
        String operation,
        Map<String, java.util.List<String>> inputs,
        Map<String, String> outputs,
        Map<String, ConfigurationValue> attributes,
        Map<String, String> extensions) {
    public OperationDescriptor {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("id must not be blank");
        if (operation == null || operation.isBlank()) throw new IllegalArgumentException("operation must not be blank");
        inputs = immutableInputs(inputs);
        outputs = immutableStringMap(outputs, "outputs");
        attributes = immutableAttributeMap(attributes);
        extensions = immutableStringMap(extensions, "extensions");
        if (outputs.isEmpty()) throw new IllegalArgumentException("operation must declare at least one output");
    }

    private static Map<String, java.util.List<String>> immutableInputs(Map<String, java.util.List<String>> values) {
        Objects.requireNonNull(values, "inputs must not be null");
        Map<String, java.util.List<String>> copy = new LinkedHashMap<>();
        values.forEach((key, value) -> {
            requireKey(key, "input port");
            java.util.List<String> ids = java.util.List.copyOf(Objects.requireNonNull(value, "input values must not be null"));
            if (ids.stream().anyMatch(id -> id == null || id.isBlank()))
                throw new IllegalArgumentException("input value ids must not be blank");
            copy.put(key, ids);
        });
        return Collections.unmodifiableMap(copy);
    }

    private static Map<String, String> immutableStringMap(Map<String, String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank() || e.getValue() == null))
            throw new IllegalArgumentException(name + " must contain non-blank keys and non-null values");
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }

    private static Map<String, ConfigurationValue> immutableAttributeMap(Map<String, ConfigurationValue> values) {
        Objects.requireNonNull(values, "attributes must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank() || e.getValue() == null))
            throw new IllegalArgumentException("attributes must contain non-blank keys and non-null values");
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }

    private static void requireKey(String key, String name) {
        if (key == null || key.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
    }
}
