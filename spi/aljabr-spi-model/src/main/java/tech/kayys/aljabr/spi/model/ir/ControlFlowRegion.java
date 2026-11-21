package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/** Explicit control-flow boundary; graph nodes are not assumed to form a DAG. */
public record ControlFlowRegion(
        String id,
        Kind kind,
        Optional<String> parentRegionId,
        List<String> nodeIds,
        List<String> inputValueIds,
        List<String> outputValueIds,
        java.util.Map<String, String> extensions) {
    public enum Kind { CONDITIONAL, LOOP, SCAN, CUSTOM }

    public ControlFlowRegion {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("id must not be blank");
        Objects.requireNonNull(kind, "kind must not be null");
        parentRegionId = Objects.requireNonNull(parentRegionId, "parentRegionId must not be null");
        parentRegionId.ifPresent(parent -> {
            if (parent.isBlank() || parent.equals(id))
                throw new IllegalArgumentException("parent region must be non-blank and not self");
        });
        nodeIds = immutableIds(nodeIds, "nodeIds");
        inputValueIds = immutableIds(inputValueIds, "inputValueIds");
        outputValueIds = immutableIds(outputValueIds, "outputValueIds");
        Objects.requireNonNull(extensions, "extensions must not be null");
        if (extensions.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank() || e.getValue() == null))
            throw new IllegalArgumentException("extensions must contain non-blank keys and non-null values");
        extensions = java.util.Collections.unmodifiableMap(new java.util.LinkedHashMap<>(extensions));
    }

    private static List<String> immutableIds(List<String> values, String name) {
        List<String> copy = List.copyOf(Objects.requireNonNull(values, name + " must not be null"));
        if (copy.stream().anyMatch(value -> value == null || value.isBlank()))
            throw new IllegalArgumentException(name + " must contain non-blank ids");
        if (copy.stream().distinct().count() != copy.size())
            throw new IllegalArgumentException(name + " must not contain duplicate ids");
        return copy;
    }
}
