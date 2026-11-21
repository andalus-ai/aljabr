package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/** Immutable tensor shape. An empty dimension list represents a scalar. */
public record TensorShape(List<Dimension> dimensions) {
    public TensorShape {
        dimensions = List.copyOf(Objects.requireNonNull(dimensions, "dimensions must not be null"));
        dimensions.forEach(dimension -> Objects.requireNonNull(dimension, "dimension must not be null"));
    }

    public static TensorShape scalar() { return new TensorShape(List.of()); }

    public static TensorShape of(long... sizes) {
        Objects.requireNonNull(sizes, "sizes must not be null");
        return new TensorShape(java.util.Arrays.stream(sizes).mapToObj(Dimension::fixed).toList());
    }

    public static TensorShape symbolic(String... symbols) {
        Objects.requireNonNull(symbols, "symbols must not be null");
        return new TensorShape(java.util.Arrays.stream(symbols).map(Dimension::symbolic).toList());
    }

    public int rank() { return dimensions.size(); }

    public boolean isStatic() { return dimensions.stream().allMatch(Dimension::isStatic); }

    /** Returns empty for symbolic shapes or when the element count overflows long. */
    public OptionalLong elementCount() {
        if (!isStatic()) return OptionalLong.empty();
        long count = 1;
        try {
            for (Dimension dimension : dimensions)
                count = Math.multiplyExact(count, dimension.fixedValue().orElseThrow());
            return OptionalLong.of(count);
        } catch (ArithmeticException overflow) {
            return OptionalLong.empty();
        }
    }

    public List<Long> staticDimensions() {
        if (!isStatic()) throw new IllegalStateException("shape contains symbolic dimensions");
        return dimensions.stream().map(d -> d.fixedValue().orElseThrow()).toList();
    }
}
