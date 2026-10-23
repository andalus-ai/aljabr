package tech.kayys.aljabr.core.tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Represents the shape (dimensions) of a tensor.
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class Shape {
    private final long[] dims;
    private final long numel;

    public Shape(long... dims) {
        Objects.requireNonNull(dims, "dims must not be null");
        this.dims = dims.clone();
        long n = 1;
        for (long d : this.dims) {
            if (d < 0) {
                throw new IllegalArgumentException("Tensor dimensions must be non-negative: " + d);
            }
            try {
                n = Math.multiplyExact(n, d);
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("Tensor element count overflows long for shape " + Arrays.toString(this.dims), e);
            }
        }
        this.numel = n;
    }

    public long[] dims() {
        return dims.clone();
    }

    public int rank() {
        return dims.length;
    }

    public long dim(int i) {
        return dims[i];
    }

    public long numel() {
        return numel;
    }

    @Override
    public String toString() {
        return Arrays.toString(dims);
    }
}