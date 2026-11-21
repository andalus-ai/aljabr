package tech.kayys.aljabr.spi.model.ir;

import java.util.Objects;

/** A typed value in a computation graph, including graph inputs and constants. */
public record GraphValue(String id, TensorType type, String description) {
    public GraphValue {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("id must not be blank");
        Objects.requireNonNull(type, "type must not be null");
        description = description == null ? "" : description;
    }

    public GraphValue(String id, TensorType type) { this(id, type, ""); }
}
