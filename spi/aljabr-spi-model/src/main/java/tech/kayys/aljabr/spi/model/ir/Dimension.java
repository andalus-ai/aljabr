package tech.kayys.aljabr.spi.model.ir;

import java.util.Objects;
import java.util.OptionalLong;

/** One tensor dimension, either statically sized or symbolically constrained. */
public record Dimension(OptionalLong fixedValue, String symbol, OptionalLong minimum, OptionalLong maximum) {
    public Dimension {
        Objects.requireNonNull(fixedValue, "fixedValue must not be null");
        Objects.requireNonNull(minimum, "minimum must not be null");
        Objects.requireNonNull(maximum, "maximum must not be null");
        if (fixedValue.isPresent() && fixedValue.getAsLong() < 0)
            throw new IllegalArgumentException("dimension size must not be negative");
        if (minimum.isPresent() && minimum.getAsLong() < 0)
            throw new IllegalArgumentException("minimum must not be negative");
        if (maximum.isPresent() && maximum.getAsLong() < 0)
            throw new IllegalArgumentException("maximum must not be negative");
        if (minimum.isPresent() && maximum.isPresent() && minimum.getAsLong() > maximum.getAsLong())
            throw new IllegalArgumentException("minimum must not exceed maximum");
        if (fixedValue.isPresent()) {
            long value = fixedValue.getAsLong();
            if ((minimum.isPresent() && value < minimum.getAsLong())
                    || (maximum.isPresent() && value > maximum.getAsLong()))
                throw new IllegalArgumentException("fixed dimension violates its bounds");
        } else if (symbol == null || symbol.isBlank()) {
            throw new IllegalArgumentException("symbolic dimensions require a non-blank symbol");
        }
    }

    public static Dimension fixed(long size) {
        return new Dimension(OptionalLong.of(size), null, OptionalLong.empty(), OptionalLong.empty());
    }

    public static Dimension symbolic(String symbol) {
        return new Dimension(OptionalLong.empty(), symbol, OptionalLong.empty(), OptionalLong.empty());
    }

    public static Dimension symbolic(String symbol, long minimum, long maximum) {
        return new Dimension(OptionalLong.empty(), symbol, OptionalLong.of(minimum), OptionalLong.of(maximum));
    }

    public boolean isStatic() { return fixedValue.isPresent(); }
}
