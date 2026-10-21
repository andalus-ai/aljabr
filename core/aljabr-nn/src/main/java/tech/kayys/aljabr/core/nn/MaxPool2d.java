package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * 2D max pooling layer.
 * @author bhangun
 */
public class MaxPool2d extends NNModule {

    private final int kernelSize;
    private final int stride;
    private final int padding;

    public MaxPool2d(int kernelSize, int stride, int padding) {
        this.kernelSize = kernelSize;
        this.stride = stride > 0 ? stride : kernelSize;
        this.padding = padding;
    }

    public MaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize, 0);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.maxPool2d(kernelSize, stride, padding);
    }

    public int kernelSize() {
        return kernelSize;
    }

    public int stride() {
        return stride;
    }

    public int padding() {
        return padding;
    }

    @Override
    public String toString() {
        return String.format("MaxPool2d(kernel_size=%d, stride=%d, padding=%d)",
                kernelSize, stride, padding);
    }
}
