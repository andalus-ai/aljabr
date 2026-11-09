package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/** 2D max pooling layer. */
public class MaxPool2d extends NNModule {

    private final int kernelSize;
    private final int stride;
    private final int padding;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public MaxPool2d(int kernelSize, int stride, int padding) {
        this(null, kernelSize, stride, padding);
    }

    public MaxPool2d(ComputeBackend backend, int kernelSize, int stride, int padding) {
        super(backend);
        if (kernelSize <= 0 || padding < 0) {
            throw new IllegalArgumentException("kernelSize must be positive and padding non-negative");
        }
        this.kernelSize = kernelSize;
        this.stride = stride > 0 ? stride : kernelSize;
        this.padding = padding;
    }

    /** @deprecated Use {@link #MaxPool2d(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public MaxPool2d(int kernelSize) {
        this(null, kernelSize, kernelSize, 0);
    }

    public MaxPool2d(ComputeBackend backend, int kernelSize) {
        this(backend, kernelSize, kernelSize, 0);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input.maxPool2d(kernelSize, stride, padding);
    }

    public int kernelSize() { return kernelSize; }
    public int stride() { return stride; }
    public int padding() { return padding; }

    @Override
    public String toString() {
        return String.format("MaxPool2d(kernel_size=%d, stride=%d, padding=%d)",
                kernelSize, stride, padding);
    }
}
