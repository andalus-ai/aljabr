package tech.kayys.aljabr.spi.model.ir;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/** Format-neutral quantization metadata; scheme identifiers remain extensible. */
public record QuantizationDescriptor(
        String scheme,
        TensorDataType storageType,
        OptionalInt axis,
        Optional<BigDecimal> scale,
        OptionalLong zeroPoint,
        Map<String, String> extensions) {
    public QuantizationDescriptor {
        if (scheme == null || scheme.isBlank()) throw new IllegalArgumentException("scheme must not be blank");
        Objects.requireNonNull(storageType, "storageType must not be null");
        Objects.requireNonNull(axis, "axis must not be null");
        Objects.requireNonNull(scale, "scale must not be null");
        Objects.requireNonNull(zeroPoint, "zeroPoint must not be null");
        if (axis.isPresent() && axis.getAsInt() < 0) throw new IllegalArgumentException("axis must not be negative");
        if (scale.isPresent() && scale.orElseThrow().signum() <= 0)
            throw new IllegalArgumentException("scale must be positive");
        Objects.requireNonNull(extensions, "extensions must not be null");
        if (extensions.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank() || e.getValue() == null))
            throw new IllegalArgumentException("extensions must have non-blank keys and non-null values");
        extensions = Collections.unmodifiableMap(new LinkedHashMap<>(extensions));
    }

    public static QuantizationDescriptor scheme(String scheme, TensorDataType storageType) {
        return new QuantizationDescriptor(scheme, storageType, OptionalInt.empty(),
                Optional.empty(), OptionalLong.empty(), Map.of());
    }
}
