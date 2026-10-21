package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Adaptive average pooling 2D layer.
 * @author bhangun
 */
public class AdaptiveAvgPool2d extends NNModule {

    private final int outputH;
    private final int outputW;

    public AdaptiveAvgPool2d(int outputH, int outputW) {
        this.outputH = outputH;
        this.outputW = outputW;
    }

    public AdaptiveAvgPool2d(int outputSize) {
        this(outputSize, outputSize);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.adaptiveAvgPool2d(outputH, outputW);
    }

    public int outputH() {
        return outputH;
    }

    public int outputW() {
        return outputW;
    }

    @Override
    public String toString() {
        return String.format("AdaptiveAvgPool2d(output_size=(%d, %d))", outputH, outputW);
    }
}
