package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;

/**
 * Format-neutral description of one model tensor.
 *
 * @param name stable tensor name used by the model architecture
 * @param dataType canonical or source-defined data type identifier
 * @param shape tensor dimensions, with scalars represented by an empty list
 * @param layout logical layout identifier, or {@code unknown} when unavailable
 * @param quantization quantization scheme identifier, or {@code null} when not quantized/unknown
 * @param payload optional reference to the original tensor bytes
 * @param extensions opaque source-specific attributes that must not be silently discarded
 */
public record TensorDescriptor(
        String name,
        String dataType,
        List<Long> shape,
        String layout,
        String quantization,
        PayloadReference payload,
        java.util.Map<String, String> extensions) {

    public TensorDescriptor {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (dataType == null || dataType.isBlank()) {
            throw new IllegalArgumentException("dataType must not be blank");
        }
        shape = List.copyOf(Objects.requireNonNull(shape, "shape must not be null"));
        if (shape.stream().anyMatch(dimension -> dimension == null || dimension < 0L)) {
            throw new IllegalArgumentException("shape dimensions must be non-negative");
        }
        layout = layout == null || layout.isBlank() ? "unknown" : layout;
        extensions = java.util.Collections.unmodifiableMap(
                new java.util.LinkedHashMap<>(Objects.requireNonNull(extensions, "extensions must not be null")));
    }

    public TensorDescriptor(
            String name,
            String dataType,
            List<Long> shape,
            String layout,
            String quantization,
            PayloadReference payload) {
        this(name, dataType, shape, layout, quantization, payload, java.util.Map.of());
    }
}
