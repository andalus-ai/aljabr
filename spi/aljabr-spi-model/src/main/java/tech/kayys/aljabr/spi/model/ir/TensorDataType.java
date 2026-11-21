package tech.kayys.aljabr.spi.model.ir;

import java.util.Objects;

/** Canonical, extensible tensor element type. */
public record TensorDataType(String id, Category category, int bitWidth) {
    public enum Category { FLOATING_POINT, SIGNED_INTEGER, UNSIGNED_INTEGER, BOOLEAN, COMPLEX, OPAQUE }

    public static final TensorDataType F16 = of("f16", Category.FLOATING_POINT, 16);
    public static final TensorDataType BF16 = of("bf16", Category.FLOATING_POINT, 16);
    public static final TensorDataType F32 = of("f32", Category.FLOATING_POINT, 32);
    public static final TensorDataType F64 = of("f64", Category.FLOATING_POINT, 64);
    public static final TensorDataType I8 = of("i8", Category.SIGNED_INTEGER, 8);
    public static final TensorDataType I16 = of("i16", Category.SIGNED_INTEGER, 16);
    public static final TensorDataType I32 = of("i32", Category.SIGNED_INTEGER, 32);
    public static final TensorDataType I64 = of("i64", Category.SIGNED_INTEGER, 64);
    public static final TensorDataType U8 = of("u8", Category.UNSIGNED_INTEGER, 8);
    public static final TensorDataType U16 = of("u16", Category.UNSIGNED_INTEGER, 16);
    public static final TensorDataType U32 = of("u32", Category.UNSIGNED_INTEGER, 32);
    public static final TensorDataType U64 = of("u64", Category.UNSIGNED_INTEGER, 64);
    public static final TensorDataType BOOL = of("bool", Category.BOOLEAN, 8);
    public static final TensorDataType C64 = of("c64", Category.COMPLEX, 64);
    public static final TensorDataType C128 = of("c128", Category.COMPLEX, 128);

    public TensorDataType {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("id must not be blank");
        Objects.requireNonNull(category, "category must not be null");
        if (bitWidth < 0) throw new IllegalArgumentException("bitWidth must not be negative");
        if (category != Category.OPAQUE && bitWidth == 0)
            throw new IllegalArgumentException("known numeric types must have a positive bitWidth");
    }

    public static TensorDataType of(String id, Category category, int bitWidth) {
        return new TensorDataType(id, category, bitWidth);
    }

    public static TensorDataType opaque(String id) {
        return new TensorDataType(id, Category.OPAQUE, 0);
    }
}
