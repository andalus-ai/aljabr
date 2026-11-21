package tech.kayys.aljabr.spi.model.ir;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Immutable computation graph. Cycles are permitted for recurrent and control-flow graphs. */
public record GraphDescriptor(
        String id,
        List<GraphValue> values,
        List<OperationDescriptor> operations,
        GraphSignature signature,
        List<ControlFlowRegion> controlFlowRegions,
        Map<String, String> extensions) {
    public GraphDescriptor {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("id must not be blank");
        values = List.copyOf(Objects.requireNonNull(values, "values must not be null"));
        operations = List.copyOf(Objects.requireNonNull(operations, "operations must not be null"));
        Objects.requireNonNull(signature, "signature must not be null");
        controlFlowRegions = List.copyOf(Objects.requireNonNull(controlFlowRegions, "controlFlowRegions must not be null"));
        extensions = immutableExtensions(extensions);

        Map<String, GraphValue> valuesById = new HashMap<>();
        for (GraphValue value : values) {
            Objects.requireNonNull(value, "graph value must not be null");
            if (valuesById.putIfAbsent(value.id(), value) != null)
                throw new IllegalArgumentException("duplicate graph value id: " + value.id());
        }

        Set<String> operationIds = new HashSet<>();
        Set<String> producedValues = new HashSet<>();
        for (OperationDescriptor operation : operations) {
            Objects.requireNonNull(operation, "operation must not be null");
            if (!operationIds.add(operation.id()))
                throw new IllegalArgumentException("duplicate operation id: " + operation.id());
            operation.inputs().values().forEach(ids -> ids.forEach(valueId -> requireValue(valuesById, valueId)));
            operation.outputs().forEach((port, valueId) -> {
                requireValue(valuesById, valueId);
                if (!producedValues.add(valueId))
                    throw new IllegalArgumentException("graph value has multiple producers: " + valueId);
            });
        }

        signature.inputs().forEach((name, valueId) -> requireValue(valuesById, valueId));
        signature.outputs().forEach((name, valueId) -> requireValue(valuesById, valueId));

        Map<String, ControlFlowRegion> regionsById = new HashMap<>();
        for (ControlFlowRegion region : controlFlowRegions) {
            Objects.requireNonNull(region, "control-flow region must not be null");
            if (regionsById.putIfAbsent(region.id(), region) != null)
                throw new IllegalArgumentException("duplicate control-flow region id: " + region.id());
            region.nodeIds().forEach(nodeId -> {
                if (!operationIds.contains(nodeId))
                    throw new IllegalArgumentException("region references unknown operation: " + nodeId);
            });
            region.inputValueIds().forEach(valueId -> requireValue(valuesById, valueId));
            region.outputValueIds().forEach(valueId -> requireValue(valuesById, valueId));
        }
        for (ControlFlowRegion region : controlFlowRegions) {
            region.parentRegionId().ifPresent(parent -> {
                if (!regionsById.containsKey(parent))
                    throw new IllegalArgumentException("unknown parent control-flow region: " + parent);
            });
            Set<String> ancestors = new HashSet<>();
            ControlFlowRegion current = region;
            while (current.parentRegionId().isPresent()) {
                String parentId = current.parentRegionId().orElseThrow();
                if (!ancestors.add(parentId))
                    throw new IllegalArgumentException("cyclic control-flow region nesting: " + region.id());
                current = regionsById.get(parentId);
            }
        }
    }

    public static GraphDescriptor of(String id, List<GraphValue> values, List<OperationDescriptor> operations,
            GraphSignature signature) {
        return new GraphDescriptor(id, values, operations, signature, List.of(), Map.of());
    }

    private static void requireValue(Map<String, GraphValue> values, String valueId) {
        if (!values.containsKey(valueId))
            throw new IllegalArgumentException("graph references unknown value: " + valueId);
    }

    private static Map<String, String> immutableExtensions(Map<String, String> values) {
        Objects.requireNonNull(values, "extensions must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank() || e.getValue() == null))
            throw new IllegalArgumentException("extensions must contain non-blank keys and non-null values");
        return java.util.Collections.unmodifiableMap(new java.util.LinkedHashMap<>(values));
    }
}
