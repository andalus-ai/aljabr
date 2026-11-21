package tech.kayys.aljabr.spi.model.ir;

import java.util.Objects;
import java.util.Optional;

/** Complete logical tensor type independent of payload storage or backend. */
public record TensorType(
        TensorDataType dataType,
        TensorShape shape,
        TensorLayout layout,
        Optional<QuantizationDescriptor> quantization) {
    public TensorType {
        Objects.requireNonNull(dataType, "dataType must not be null");
        Objects.requireNonNull(shape, "shape must not be null");
        Objects.requireNonNull(layout, "layout must not be null");
        Objects.requireNonNull(quantization, "quantization must not be null");
        layout.validateRank(shape.rank());
        quantization.ifPresent(q -> {
            if (q.axis().isPresent() && q.axis().getAsInt() >= shape.rank())
                throw new IllegalArgumentException("quantization axis must be within tensor rank");
        });
    }

    public static TensorType of(TensorDataType dataType, TensorShape shape) {
        return new TensorType(dataType, shape, TensorLayout.UNKNOWN, Optional.empty());
    }

    public TensorType withLayout(TensorLayout replacement) {
        return new TensorType(dataType, shape, replacement, quantization);
    }

    public TensorType withQuantization(QuantizationDescriptor replacement) {
        return new TensorType(dataType, shape, layout, Optional.of(replacement));
    }
}
