package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;

/** Logical tensor layout, optionally carrying explicit element strides. */
public record TensorLayout(String id, List<Long> strides) {
    public static final TensorLayout UNKNOWN = new TensorLayout("unknown", List.of());
    public static final TensorLayout CONTIGUOUS = new TensorLayout("contiguous", List.of());

    public TensorLayout {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("id must not be blank");
        strides = List.copyOf(Objects.requireNonNull(strides, "strides must not be null"));
        if (strides.stream().anyMatch(stride -> stride == null || stride < 0))
            throw new IllegalArgumentException("strides must be non-negative");
    }

    public static TensorLayout strided(List<Long> strides) {
        return new TensorLayout("strided", strides);
    }

    public boolean hasExplicitStrides() { return !strides.isEmpty(); }

    public void validateRank(int rank) {
        if (hasExplicitStrides() && strides.size() != rank)
            throw new IllegalArgumentException("stride count must match tensor rank");
    }
}
