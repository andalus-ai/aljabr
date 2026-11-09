package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/** Adaptive average pooling 2D layer. */
public class AdaptiveAvgPool2d extends NNModule {

    private final int outputH;
    private final int outputW;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public AdaptiveAvgPool2d(int outputH, int outputW) {
        this(null, outputH, outputW);
    }

    public AdaptiveAvgPool2d(ComputeBackend backend, int outputH, int outputW) {
        super(backend);
        if (outputH <= 0 || outputW <= 0) {
            throw new IllegalArgumentException("Output dimensions must be positive");
        }
        this.outputH = outputH;
        this.outputW = outputW;
    }

    /** @deprecated Use {@link #AdaptiveAvgPool2d(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public AdaptiveAvgPool2d(int outputSize) {
        this(null, outputSize, outputSize);
    }

    public AdaptiveAvgPool2d(ComputeBackend backend, int outputSize) {
        this(backend, outputSize, outputSize);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input.adaptiveAvgPool2d(outputH, outputW);
    }

    public int outputH() { return outputH; }
    public int outputW() { return outputW; }

    @Override
    public String toString() {
        return String.format("AdaptiveAvgPool2d(output_size=(%d, %d))", outputH, outputW);
    }
}
