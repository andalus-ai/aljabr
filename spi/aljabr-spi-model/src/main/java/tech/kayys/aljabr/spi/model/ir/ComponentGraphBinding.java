package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Connects named component ports to named ports on a model graph.
 * Port names are component-defined; this descriptor does not prescribe a
 * tokenizer or modality-specific tensor convention.
 */
public record ComponentGraphBinding(
        String graphId,
        Map<String, String> inputPorts,
        Map<String, String> outputPorts) {
    public ComponentGraphBinding {
        graphId = requireText(graphId, "graphId");
        inputPorts = immutableMap(inputPorts, "inputPorts");
        outputPorts = immutableMap(outputPorts, "outputPorts");
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }

    private static Map<String, String> immutableMap(Map<String, String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank()
                || e.getValue() == null || e.getValue().isBlank())) {
            throw new IllegalArgumentException(name + " must contain non-blank keys and values");
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }
}
