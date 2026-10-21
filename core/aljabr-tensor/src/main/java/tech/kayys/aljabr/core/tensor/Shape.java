package tech.kayys.aljabr.core.tensor;

import java.util.Arrays;

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
        this.dims = dims.clone();
        long n = 1;
        for (long d : dims)
            n *= d;
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