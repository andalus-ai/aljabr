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
 * Immutable, format-neutral descriptor of a model and its tensor inventory.
 *
 * <p>This is a metadata-first representation: it describes model identity,
 * architecture, source format, metadata, extensions, and tensors without
 * requiring tensor payloads to be loaded. Payload bytes remain reachable only
 * through optional {@link PayloadReference} values on tensor descriptors.
 *
 * <p>Architecture identifiers and source-format identifiers are opaque strings
 * so AMIR can preserve models from format and architecture families that the
 * current Aljabr release does not yet understand. When known, richer typed
 * architecture information is available through {@link #architectureDescriptor()}.
 *
 * @param identity stable model identity
 * @param architecture stable architecture identifier
 * @param sourceFormat identifier of the source serialization format
 * @param metadata standardized or source-provided metadata
 * @param extensions opaque namespaced metadata not interpreted by this version
 * @param tensors tensor descriptors, with unique names
 * @param architectureDescriptor optional typed architecture details
 * @param graphCollection optional collection of model computation graphs
 */
public record ModelDescriptor(
        ModelIdentity identity,
        String architecture,
        String sourceFormat,
        Map<String, String> metadata,
        Map<String, String> extensions,
        List<TensorDescriptor> tensors,
        Optional<ArchitectureDescriptor> architectureDescriptor,
        Optional<GraphCollection> graphCollection) {

    /**
     * Compatibility constructor for descriptors created before typed
     * architecture information was introduced.
     */
    public ModelDescriptor(
            ModelIdentity identity,
            String architecture,
            String sourceFormat,
            Map<String, String> metadata,
            Map<String, String> extensions,
            List<TensorDescriptor> tensors,
            Optional<ArchitectureDescriptor> architectureDescriptor) {
        this(identity, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, Optional.empty());
    }

    /** Compatibility constructor for the original metadata-only descriptor. */
    public ModelDescriptor(
            ModelIdentity identity,
            String architecture,
            String sourceFormat,
            Map<String, String> metadata,
            Map<String, String> extensions,
            List<TensorDescriptor> tensors) {
        this(identity, architecture, sourceFormat, metadata, extensions, tensors,
                Optional.empty(), Optional.empty());
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

        if (architectureDescriptor.isPresent()
                && !architecture.equals(architectureDescriptor.orElseThrow().id())) {
            throw new IllegalArgumentException(
                    "architecture identifier must match architecture descriptor id");
        }

        Set<String> names = new HashSet<>();
        for (TensorDescriptor tensor : tensors) {
            Objects.requireNonNull(tensor, "tensor descriptor must not be null");
            if (!names.add(tensor.name())) {
                throw new IllegalArgumentException("Duplicate tensor name: " + tensor.name());
            }
        }
    }

    /** Returns a descriptor with the same content and a different identity. */
    public ModelDescriptor withIdentity(ModelIdentity replacement) {
        return new ModelDescriptor(
                replacement, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, graphCollection);
    }

    /** Returns a descriptor enriched with typed architecture information. */
    public ModelDescriptor withArchitectureDescriptor(ArchitectureDescriptor replacement) {
        Objects.requireNonNull(replacement, "replacement must not be null");
        return new ModelDescriptor(
                identity, replacement.id(), sourceFormat, metadata, extensions, tensors,
                Optional.of(replacement), graphCollection);
    }

 
    /** Returns a descriptor enriched with its computation graph collection. */
    public ModelDescriptor withGraphCollection(GraphCollection replacement) {
        Objects.requireNonNull(replacement, "replacement must not be null");
        return new ModelDescriptor(identity, architecture, sourceFormat, metadata, extensions, tensors,
                architectureDescriptor, Optional.of(replacement));
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
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
