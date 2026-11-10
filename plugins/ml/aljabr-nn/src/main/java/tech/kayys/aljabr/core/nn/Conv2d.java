package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * 2D convolution layer.
 */
public class Conv2d extends NNModule {

    private final long inChannels;
    private final long outChannels;
    private final long kernelSize;
    private final int stride;
    private final int padding;
    private final int dilation;
    private final int groups;
    private final boolean hasBias;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Conv2d(long inChannels, long outChannels, long kernelSize,
            int stride, int padding, int dilation, int groups, boolean bias) {
        this(null, inChannels, outChannels, kernelSize, stride, padding, dilation, groups, bias);
    }

    public Conv2d(ComputeBackend backend, long inChannels, long outChannels, long kernelSize,
            int stride, int padding, int dilation, int groups, boolean bias) {
        super(backend);
        if (groups <= 0 || inChannels % groups != 0 || outChannels % groups != 0) {
            throw new IllegalArgumentException("groups must be positive and divide input/output channels");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        this.hasBias = bias;

        registerParameter("weight", randomNormal(outChannels, inChannels / groups, kernelSize, kernelSize));
        if (bias) {
            registerParameter("bias", zeros(outChannels));
        }
    }

    /** @deprecated Use {@link #Conv2d(ComputeBackend, long, long, long)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Conv2d(long inChannels, long outChannels, long kernelSize) {
        this(null, inChannels, outChannels, kernelSize, 1, 0, 1, 1, true);
    }

    public Conv2d(ComputeBackend backend, long inChannels, long outChannels, long kernelSize) {
        this(backend, inChannels, outChannels, kernelSize, 1, 0, 1, 1, true);
    }

    public Conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        super(Objects.requireNonNull(weight, "weight must not be null").backend());
        this.outChannels = weight.shape().dim(0);
        this.inChannels = weight.shape().dim(1) * groups;
        this.kernelSize = weight.shape().dim(2);
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        this.hasBias = (bias != null);
        registerParameter("weight", weight);
        if (bias != null) {
            registerParameter("bias", bias);
        }
    }

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Conv2d(long inChannels, long outChannels, long kernelSize, int stride, int padding) {
        this(null, inChannels, outChannels, kernelSize, stride, padding, 1, 1, true);
    }

    public Conv2d(ComputeBackend backend, long inChannels, long outChannels,
            long kernelSize, int stride, int padding) {
        this(backend, inChannels, outChannels, kernelSize, stride, padding, 1, 1, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        Tensor weight = parameters.get("weight");
        Tensor bias = hasBias ? parameters.get("bias") : null;
        return input.conv2d(weight, bias, stride, padding, dilation, groups);
    }

    public long inChannels() { return inChannels; }
    public long outChannels() { return outChannels; }
    public long kernelSize() { return kernelSize; }
    public int stride() { return stride; }
    public int padding() { return padding; }
    public int dilation() { return dilation; }
    public int groups() { return groups; }
    public boolean hasBias() { return hasBias; }

    @Override
    public String toString() {
        return String.format("Conv2d(%d, %d, kernel_size=%d, stride=%d, padding=%d)",
                inChannels, outChannels, kernelSize, stride, padding);
    }
}
