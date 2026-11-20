package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Format-neutral description of one model tensor.
 *
 * <p>The legacy string/list fields remain available for existing adapters.
 * New code should use {@link #tensorType()} when present; it can represent
 * symbolic dimensions, explicit strides, and structured quantization.
 *
 * @param name stable tensor name used by the model architecture
 * @param dataType canonical or source-defined data type identifier
 * @param shape legacy static dimensions, with scalars represented by an empty list
 * @param layout logical layout identifier, or {@code unknown} when unavailable
 * @param quantization legacy quantization scheme identifier
 * @param payload optional reference to the original tensor bytes
 * @param extensions opaque source-specific attributes
 * @param tensorType optional canonical typed tensor information
 */
public record TensorDescriptor(
        String name,
        String dataType,
        List<Long> shape,
        String layout,
        String quantization,
        PayloadReference payload,
        java.util.Map<String, String> extensions,
        Optional<TensorType> tensorType) {

    /** Compatibility constructor retaining the original seven-argument API. */
    public TensorDescriptor(
            String name, String dataType, List<Long> shape, String layout,
            String quantization, PayloadReference payload, java.util.Map<String, String> extensions) {
        this(name, dataType, shape, layout, quantization, payload, extensions, Optional.empty());
    }

    /** Compatibility constructor retaining the original six-argument API. */
    public TensorDescriptor(
            String name, String dataType, List<Long> shape, String layout,
            String quantization, PayloadReference payload) {
        this(name, dataType, shape, layout, quantization, payload, java.util.Map.of(), Optional.empty());
    }

    public TensorDescriptor {
        if (name == null || name.isBlank()) throw new IllegalArgumentException("name must not be blank");
        if (dataType == null || dataType.isBlank()) throw new IllegalArgumentException("dataType must not be blank");
        shape = List.copyOf(Objects.requireNonNull(shape, "shape must not be null"));
        if (shape.stream().anyMatch(dimension -> dimension == null || dimension < 0L))
            throw new IllegalArgumentException("shape dimensions must be non-negative");
        layout = layout == null || layout.isBlank() ? "unknown" : layout;
        extensions = java.util.Collections.unmodifiableMap(
                new java.util.LinkedHashMap<>(Objects.requireNonNull(extensions, "extensions must not be null")));
        tensorType = Objects.requireNonNull(tensorType, "tensorType must not be null");
        tensorType.ifPresent(type -> {
            if (!dataType.equals(type.dataType().id()))
                throw new IllegalArgumentException("legacy dataType must match typed tensor data type");
            if (type.shape().isStatic() && !shape.equals(type.shape().staticDimensions()))
                throw new IllegalArgumentException("legacy shape must match typed static shape");
            if (!"unknown".equals(layout) && !layout.equals(type.layout().id()))
                throw new IllegalArgumentException("legacy layout must match typed tensor layout");
            if (quantization != null && type.quantization().isPresent()
                    && !quantization.equals(type.quantization().orElseThrow().scheme()))
                throw new IllegalArgumentException("legacy quantization must match typed quantization scheme");
        });
    }

    /** Creates a descriptor whose canonical type is represented by the typed IR. */
    public static TensorDescriptor typed(
            String name, TensorType type, PayloadReference payload, java.util.Map<String, String> extensions) {
        Objects.requireNonNull(type, "type must not be null");
        return new TensorDescriptor(
                name,
                type.dataType().id(),
                type.shape().isStatic() ? type.shape().staticDimensions() : List.of(),
                type.layout().id(),
                type.quantization().map(QuantizationDescriptor::scheme).orElse(null),
                payload,
                extensions,
                Optional.of(type));
    }

    /** Returns this descriptor enriched with canonical type information. */
    public TensorDescriptor withTensorType(TensorType replacement) {
        return typed(name, replacement, payload, extensions);
    }
}
