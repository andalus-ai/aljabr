package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * 2D convolution layer.
 * @author bhangun
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

    public Conv2d(long inChannels, long outChannels, long kernelSize,
            int stride, int padding, int dilation, int groups, boolean bias) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        this.hasBias = bias;

        // Weight shape: [outChannels, inChannels/groups, kernelSize, kernelSize]
        Tensor weight = Tensor.randn(outChannels, inChannels / groups, kernelSize, kernelSize);
        registerParameter("weight", weight);

        if (bias) {
            Tensor biasParam = Tensor.zeros(outChannels);
            registerParameter("bias", biasParam);
        }
    }

    public Conv2d(long inChannels, long outChannels, long kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, 1, 1, true);
    }

    public Conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
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

    public Conv2d(long inChannels, long outChannels, long kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, 1, 1, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = parameters.get("weight");
        Tensor bias = hasBias ? parameters.get("bias") : null;
        return input.conv2d(weight, bias, stride, padding, dilation, groups);
    }

    public long inChannels() {
        return inChannels;
    }

    public long outChannels() {
        return outChannels;
    }

    public long kernelSize() {
        return kernelSize;
    }

    public int stride() {
        return stride;
    }

    public int padding() {
        return padding;
    }

    public int dilation() {
        return dilation;
    }

    public int groups() {
        return groups;
    }

    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public String toString() {
        return String.format("Conv2d(%d, %d, kernel_size=%d, stride=%d, padding=%d)",
                inChannels, outChannels, kernelSize, stride, padding);
    }
}
