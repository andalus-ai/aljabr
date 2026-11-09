package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Learned 2D spatial downsampling using a 3x3 convolution with stride 2
 * and padding 1.
 *
 * <p>Inputs and outputs use NCHW layout. Channel projection is optional:
 * when {@code outChannels} differs from {@code inChannels}, the convolution
 * performs the projection while reducing spatial resolution.</p>
 */
public final class Downsample2d extends NNModule {

    private final int inChannels;
    private final int outChannels;
    private final Conv2d convolution;

    public Downsample2d(ComputeBackend backend, int channels) {
        this(backend, channels, channels);
    }

    public Downsample2d(ComputeBackend backend, int inChannels, int outChannels) {
        super(backend);
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("channel dimensions must be positive");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.convolution = register("convolution",
                new Conv2d(backend, inChannels, outChannels, 3, 2, 1));
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        if (input.shape().rank() != 4 || input.shape().dim(1) != inChannels) {
            throw new IllegalArgumentException(
                    "Downsample2d expects NCHW input with " + inChannels + " channels");
        }
        if (input.shape().dim(2) < 2 || input.shape().dim(3) < 2) {
            throw new IllegalArgumentException("spatial dimensions must be at least 2");
        }
        return convolution.forward(input);
    }

    public int inChannels() {
        return inChannels;
    }

    public int outChannels() {
        return outChannels;
    }

    public Conv2d convolution() {
        return convolution;
    }
}
