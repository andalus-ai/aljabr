package tech.kayys.aljabr.gguf.api;

import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Structural requirement for a named GGUF tensor.
 *
 * <p>An empty dimensions list means any shape is accepted. An empty accepted
 * type set means any GGML tensor type is accepted. Type identifiers are the
 * stable numeric identifiers defined by the GGUF/ggml format.</p>
 */
public record GgufTensorRequirement(
        String name,
        List<Long> dimensions,
        Set<Integer> acceptedGgmlTypeIds) {

    public GgufTensorRequirement {
        name = requireText(name, "name");
        dimensions = List.copyOf(Objects.requireNonNull(dimensions, "dimensions"));
        for (Long dimension : dimensions) {
            Objects.requireNonNull(dimension, "dimension");
            if (dimension <= 0) {
                throw new IllegalArgumentException("required dimensions must be positive");
            }
        }
        TreeSet<Integer> types = new TreeSet<>();
        for (Integer typeId : Objects.requireNonNull(acceptedGgmlTypeIds, "acceptedGgmlTypeIds")) {
            Objects.requireNonNull(typeId, "accepted GGML type id");
            if (typeId < 0) {
                throw new IllegalArgumentException("accepted GGML type ids must not be negative");
            }
            types.add(typeId);
        }
        acceptedGgmlTypeIds = java.util.Collections.unmodifiableSet(types);
    }

    public static GgufTensorRequirement named(String name) {
        return new GgufTensorRequirement(name, List.of(), Set.of());
    }

    public static GgufTensorRequirement withShape(String name, long... dimensions) {
        Objects.requireNonNull(dimensions, "dimensions");
        return new GgufTensorRequirement(name,
                java.util.Arrays.stream(dimensions).boxed().toList(), Set.of());
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        String normalized = value.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return normalized;
    }
}
