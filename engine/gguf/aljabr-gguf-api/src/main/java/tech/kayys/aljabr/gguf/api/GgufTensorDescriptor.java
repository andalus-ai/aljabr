package tech.kayys.aljabr.gguf.api;

import java.util.List;
import java.util.Objects;

/**
 * Public structural facts for one tensor in a GGUF artifact.
 *
 * <p>This is the stable inspection/selection contract: it contains no
 * parser-specific file offsets, payload sizes, backend types, or live memory
 * handles. Parser and runtime representations are adapted to this descriptor
 * at their respective boundaries.
 */
public record GgufTensorDescriptor(String name, List<Long> dimensions, int ggmlTypeId) {

    public GgufTensorDescriptor {
        name = requireText(name, "name");
        dimensions = List.copyOf(Objects.requireNonNull(dimensions, "dimensions"));
        if (dimensions.isEmpty()) {
            throw new IllegalArgumentException("dimensions must not be empty");
        }
        for (Long dimension : dimensions) {
            Objects.requireNonNull(dimension, "dimension");
            if (dimension <= 0) {
                throw new IllegalArgumentException("tensor dimensions must be positive");
            }
        }
        if (ggmlTypeId < 0) {
            throw new IllegalArgumentException("ggmlTypeId must not be negative");
        }
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
