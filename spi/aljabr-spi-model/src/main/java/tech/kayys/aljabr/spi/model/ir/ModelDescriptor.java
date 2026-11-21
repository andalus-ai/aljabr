package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Immutable, format-neutral descriptor of a model and its metadata-first
 * structure. Tensor payloads and external assets are referenced, not loaded.
 *
 * <p>Architecture, graph, component, and asset descriptors are independent
 * typed layers. Source-format and architecture identifiers remain opaque so
 * unknown ecosystems can be preserved without coercion.
 *
 * @param identity stable model identity
 * @param architecture stable architecture identifier
 * @param sourceFormat source serialization format identifier
 * @param metadata standardized or source-provided metadata
 * @param extensions opaque namespaced metadata
 * @param tensors tensor descriptors, with unique names
 * @param architectureDescriptor optional typed architecture details
 * @param graphCollection optional computation graphs
 * @param assetCollection optional external asset references
 * @param componentCollection optional tokenizer and modality components
 */
public record ModelDescriptor(
        ModelIdentity identity,
        String architecture,
        String sourceFormat,
        Map<String, String> metadata,
        Map<String, String> extensions,
        List<TensorDescriptor> tensors,
        Optional<ArchitectureDescriptor> architectureDescriptor,
        Optional<GraphCollection> graphCollection,
        Optional<AssetCollection> assetCollection,
        Optional<ComponentCollection> componentCollection) {

    /** Compatibility constructor retaining the graph-era descriptor shape. */
    public ModelDescriptor(
            ModelIdentity identity, String architecture, String sourceFormat,
            Map<String, String> metadata, Map<String, String> extensions,
            List<TensorDescriptor> tensors,
            Optional<ArchitectureDescriptor> architectureDescriptor,
            Optional<GraphCollection> graphCollection) {
        this(identity, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, graphCollection, Optional.empty(), Optional.empty());
    }

    /** Compatibility constructor retaining the typed-architecture descriptor shape. */
    public ModelDescriptor(
            ModelIdentity identity, String architecture, String sourceFormat,
            Map<String, String> metadata, Map<String, String> extensions,
            List<TensorDescriptor> tensors,
            Optional<ArchitectureDescriptor> architectureDescriptor) {
        this(identity, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, Optional.empty(), Optional.empty(), Optional.empty());
    }

    /** Compatibility constructor for the original metadata-only descriptor. */
    public ModelDescriptor(
            ModelIdentity identity, String architecture, String sourceFormat,
            Map<String, String> metadata, Map<String, String> extensions,
            List<TensorDescriptor> tensors) {
        this(identity, architecture, sourceFormat, metadata, extensions, tensors,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public ModelDescriptor {
        Objects.requireNonNull(identity, "identity must not be null");
        architecture = requireText(architecture, "architecture");
        sourceFormat = requireText(sourceFormat, "sourceFormat");
        metadata = immutableMap(metadata, "metadata");
        extensions = immutableMap(extensions, "extensions");
        tensors = List.copyOf(Objects.requireNonNull(tensors, "tensors must not be null"));
        architectureDescriptor = Objects.requireNonNull(
                architectureDescriptor, "architectureDescriptor must not be null");
        graphCollection = Objects.requireNonNull(graphCollection, "graphCollection must not be null");
        assetCollection = Objects.requireNonNull(assetCollection, "assetCollection must not be null");
        componentCollection = Objects.requireNonNull(componentCollection, "componentCollection must not be null");

        if (architectureDescriptor.isPresent()
                && !architecture.equals(architectureDescriptor.orElseThrow().id())) {
            throw new IllegalArgumentException(
                    "architecture identifier must match architecture descriptor id");
        }

        Set<String> tensorNames = new HashSet<>();
        for (TensorDescriptor tensor : tensors) {
            Objects.requireNonNull(tensor, "tensor descriptor must not be null");
            if (!tensorNames.add(tensor.name())) {
                throw new IllegalArgumentException("Duplicate tensor name: " + tensor.name());
            }
        }

        if (componentCollection.isPresent()) {
            Set<String> assetIds = assetCollection.map(collection -> {
                Set<String> ids = new HashSet<>();
                collection.assets().forEach(asset -> ids.add(asset.id()));
                return ids;
            }).orElseGet(Set::of);
            Set<String> graphIds = graphCollection.map(collection -> {
                Set<String> ids = new HashSet<>();
                collection.graphs().forEach(graph -> ids.add(graph.id()));
                return ids;
            }).orElseGet(Set::of);

            for (ModelComponent component : componentCollection.orElseThrow().components()) {
                for (String assetId : component.assetIds()) {
                    if (!assetIds.contains(assetId)) {
                        throw new IllegalArgumentException(
                                "component '" + component.id() + "' references unknown asset: " + assetId);
                    }
                }
                for (ComponentGraphBinding binding : component.graphBindings()) {
                    if (!graphIds.contains(binding.graphId())) {
                        throw new IllegalArgumentException(
                                "component '" + component.id() + "' references unknown graph: " + binding.graphId());
                    }
                    GraphDescriptor graph = graphCollection.orElseThrow().graphs().stream()
                            .filter(candidate -> candidate.id().equals(binding.graphId()))
                            .findFirst().orElseThrow();
                    binding.inputPorts().forEach((componentPort, graphPort) -> {
                        if (!graph.signature().inputs().containsKey(graphPort)) {
                            throw new IllegalArgumentException("unknown graph input port: " + graphPort);
                        }
                    });
                    binding.outputPorts().forEach((componentPort, graphPort) -> {
                        if (!graph.signature().outputs().containsKey(graphPort)) {
                            throw new IllegalArgumentException("unknown graph output port: " + graphPort);
                        }
                    });
                }
            }
        }
    }

    /** Returns a descriptor with the same content and a different identity. */
    public ModelDescriptor withIdentity(ModelIdentity replacement) {
        return new ModelDescriptor(replacement, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, graphCollection, assetCollection, componentCollection);
    }

    /** Returns a descriptor enriched with typed architecture information. */
    public ModelDescriptor withArchitectureDescriptor(ArchitectureDescriptor replacement) {
        Objects.requireNonNull(replacement, "replacement must not be null");
        return new ModelDescriptor(identity, replacement.id(), sourceFormat, metadata, extensions, tensors,
                Optional.of(replacement), graphCollection, assetCollection, componentCollection);
    }

    /** Returns a descriptor enriched with its computation graph collection. */
    public ModelDescriptor withGraphCollection(GraphCollection replacement) {
        Objects.requireNonNull(replacement, "replacement must not be null");
        return new ModelDescriptor(identity, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, Optional.of(replacement), assetCollection, componentCollection);
    }

    /** Returns a descriptor enriched with external asset references. */
    public ModelDescriptor withAssetCollection(AssetCollection replacement) {
        Objects.requireNonNull(replacement, "replacement must not be null");
        return new ModelDescriptor(identity, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, graphCollection, Optional.of(replacement), componentCollection);
    }

    /** Returns a descriptor enriched with tokenizer and modality components. */
    public ModelDescriptor withComponentCollection(ComponentCollection replacement) {
        Objects.requireNonNull(replacement, "replacement must not be null");
        return new ModelDescriptor(identity, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, graphCollection, assetCollection, Optional.of(replacement));
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }

    private static Map<String, String> immutableMap(Map<String, String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.entrySet().stream().anyMatch(entry ->
                entry.getKey() == null || entry.getKey().isBlank() || entry.getValue() == null)) {
            throw new IllegalArgumentException(name + " must contain non-blank keys and non-null values");
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }
}
