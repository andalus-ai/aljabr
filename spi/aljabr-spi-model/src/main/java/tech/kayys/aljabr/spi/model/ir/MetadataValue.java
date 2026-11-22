package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;

/**
 * Format-neutral, lossless representation of scalar and one-dimensional model metadata.
 * Unsigned integer values use the corresponding Java wider signed type as a bit-preserving container.
 */
public sealed interface MetadataValue permits MetadataValue.Scalar, MetadataValue.Array {
    enum Kind { UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, UINT64, INT64, FLOAT64 }

    record Scalar(Kind kind, Object value) implements MetadataValue {
        public Scalar {
            Objects.requireNonNull(kind, "kind must not be null");
            Objects.requireNonNull(value, "value must not be null");
            Class<?> expected = switch (kind) {
                case UINT8, UINT16, UINT32, UINT64 -> Long.class;
                case INT8, INT16, INT32 -> Integer.class;
                case FLOAT32 -> Float.class;
                case BOOL -> Boolean.class;
                case STRING -> String.class;
                case INT64 -> Long.class;
                case FLOAT64 -> Double.class;
            };
            if (!expected.isInstance(value)) {
                throw new IllegalArgumentException("Metadata kind " + kind + " requires " + expected.getSimpleName());
            }
            long number = value instanceof Number n ? n.longValue() : 0L;
            if (kind == Kind.UINT8 && (number < 0 || number > 0xffL)
                    || kind == Kind.UINT16 && (number < 0 || number > 0xffffL)
                    || kind == Kind.UINT32 && (number < 0 || number > 0xffff_ffffL)
                    || kind == Kind.INT8 && (number < Byte.MIN_VALUE || number > Byte.MAX_VALUE)
                    || kind == Kind.INT16 && (number < Short.MIN_VALUE || number > Short.MAX_VALUE)) {
                throw new IllegalArgumentException("Metadata value is outside the range for " + kind);
            }
        }
    }

    record Array(Kind elementKind, List<Scalar> values) implements MetadataValue {
        public Array {
            Objects.requireNonNull(elementKind, "elementKind must not be null");
            if (elementKind == Kind.STRING && values == null) {
                throw new IllegalArgumentException("values must not be null");
            }
            values = List.copyOf(Objects.requireNonNull(values, "values must not be null"));
            for (Scalar value : values) {
                if (value.kind() != elementKind) {
                    throw new IllegalArgumentException("Metadata array element kind mismatch");
                }
            }
        }
    }

    static Scalar string(String value) { return new Scalar(Kind.STRING, value); }
}
