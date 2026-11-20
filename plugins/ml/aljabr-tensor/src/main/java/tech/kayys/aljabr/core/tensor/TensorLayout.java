package tech.kayys.aljabr.core.tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable logical layout of a tensor over a one-dimensional storage region.
 *
 * <p>Strides and storage offsets are measured in elements, not bytes. The
 * element width is supplied by the tensor dtype when a layout is checked
 * against a storage allocation. This descriptor deliberately does not own or
 * retain storage.</p>
 *
 * <p>Zero strides are permitted and describe overlapping/broadcast layouts.
 * Negative strides are not currently supported. A layout may be valid for
 * reading while still overlapping; mutation semantics are the responsibility
 * of the operation/backend using it.</p>
 */
public final class TensorLayout {
    private final Shape shape;
    private final long[] strides;
    private final long storageOffset;
    private final boolean contiguous;
    private final long requiredStorageElements;

    /**
     * Creates a layout with element-based strides and storage offset.
     *
     * @param shape logical tensor shape
     * @param strides one non-negative element stride per dimension
     * @param storageOffset non-negative element offset into the backing storage
     */
    public TensorLayout(Shape shape, long[] strides, long storageOffset) {
        this.shape = Objects.requireNonNull(shape, "shape must not be null");
        Objects.requireNonNull(strides, "strides must not be null");
        if (storageOffset < 0) {
            throw new IllegalArgumentException("Storage offset must be non-negative: " + storageOffset);
        }
        if (strides.length != shape.rank()) {
            throw new IllegalArgumentException(
                    "Stride rank must match shape rank: strides=" + strides.length + ", shape=" + shape.rank());
        }
        this.strides = strides.clone();
        for (int i = 0; i < this.strides.length; i++) {
            if (this.strides[i] < 0) {
                throw new IllegalArgumentException(
                        "Negative strides are not supported: dimension=" + i + ", stride=" + this.strides[i]);
            }
        }
        this.storageOffset = storageOffset;
        this.contiguous = computeContiguous(shape, this.strides);
        this.requiredStorageElements = computeRequiredStorageElements(shape, this.strides, storageOffset);
    }

    /** Returns the canonical row-major layout for {@code shape}. */
    public static TensorLayout contiguous(Shape shape) {
        Objects.requireNonNull(shape, "shape must not be null");
        long[] dims = shape.dims();
        long[] strides = new long[dims.length];
        long stride = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = Math.multiplyExact(stride, dims[i]);
        }
        return new TensorLayout(shape, strides, 0);
    }

    public Shape shape() {
        return shape;
    }

    /** Returns a defensive copy of element-based strides. */
    public long[] strides() {
        return strides.clone();
    }

    /** Returns the storage offset in elements. */
    public long storageOffset() {
        return storageOffset;
    }

    /** Returns whether this layout is contiguous in canonical row-major order. */
    public boolean isContiguous() {
        return contiguous;
    }

    /**
     * Returns the minimum storage capacity in elements needed to address this
     * layout. For an empty tensor this is its storage offset; otherwise this is
     * one past the greatest reachable element index.
     */
    public long requiredStorageElements() {
        return requiredStorageElements;
    }

    private static boolean computeContiguous(Shape shape, long[] strides) {
        if (shape.numel() == 0) {
            return true;
        }
        long expected = 1;
        for (int i = shape.rank() - 1; i >= 0; i--) {
            long dimension = shape.dim(i);
            if (dimension > 1 && strides[i] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, dimension);
        }
        return true;
    }

    private static long computeRequiredStorageElements(Shape shape, long[] strides, long offset) {
        if (shape.numel() == 0) {
            return offset;
        }
        long greatestIndex = offset;
        try {
            for (int i = 0; i < shape.rank(); i++) {
                greatestIndex = Math.addExact(
                        greatestIndex,
                        Math.multiplyExact(shape.dim(i) - 1, strides[i]));
            }
            return Math.addExact(greatestIndex, 1);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Tensor layout address range overflows long", e);
        }
    }

    @Override
    public boolean equals(Object other) {
        return this == other || other instanceof TensorLayout layout
                && shape.equals(layout.shape)
                && Arrays.equals(strides, layout.strides)
                && storageOffset == layout.storageOffset;
    }

    @Override
    public int hashCode() {
        int result = shape.hashCode();
        result = 31 * result + Arrays.hashCode(strides);
        result = 31 * result + Long.hashCode(storageOffset);
        return result;
    }

    @Override
    public String toString() {
        return "TensorLayout{shape=" + shape
                + ", strides=" + Arrays.toString(strides)
                + ", storageOffset=" + storageOffset
                + ", contiguous=" + contiguous + '}';
    }
}
