package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * One-dimensional convolution layer.
 */
public class Conv1d extends NNModule {
    private final long inChannels;
    private final long outChannels;
    private final long kernelSize;
    private final int stride;
    private final int padding;
    private final boolean hasBias;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Conv1d(long inChannels, long outChannels, long kernelSize, int stride, int padding) {
        this(null, inChannels, outChannels, kernelSize, stride, padding);
    }

    public Conv1d(ComputeBackend backend, long inChannels, long outChannels,
            long kernelSize, int stride, int padding) {
        super(backend);
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.hasBias = true;

        registerParameter("weight", randomNormal(outChannels, inChannels, kernelSize));
        registerParameter("bias", zeros(outChannels));
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        // Placeholder implementation for 1D convolution. Keep output allocation
        // on the same backend as the module parameters.
        long batch = input.shape().dim(0);
        long length = input.shape().dim(2);
        long outLength = (length + 2 * padding - kernelSize) / stride + 1;
        return zeros(batch, outChannels, outLength);
    }
}
