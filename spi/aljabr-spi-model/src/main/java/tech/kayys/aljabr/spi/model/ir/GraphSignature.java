package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Named external inputs and outputs of a graph. */
public record GraphSignature(Map<String, String> inputs, Map<String, String> outputs) {
    public GraphSignature {
        inputs = immutableBindings(inputs, "inputs");
        outputs = immutableBindings(outputs, "outputs");
    }

    public static GraphSignature empty() { return new GraphSignature(Map.of(), Map.of()); }

    private static Map<String, String> immutableBindings(Map<String, String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank()
                || e.getValue() == null || e.getValue().isBlank()))
            throw new IllegalArgumentException(name + " must contain non-blank names and value ids");
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }
}
