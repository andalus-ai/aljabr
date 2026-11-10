package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.Collection;

/**
 * Gradient clipping utility to prevent exploding gradients.
 * @author bhangun
 */
public class GradientClipper {

    public enum ClipMode {
        BY_VALUE, BY_NORM
    }

    private final ClipMode mode;
    private final double maxValue;
    private final double maxNorm;

    private double lastGlobalNorm;
    private double lastScale;

    public GradientClipper(double maxValue) {
        this(ClipMode.BY_VALUE, maxValue, maxValue);
    }

    public GradientClipper(ClipMode mode, double maxValue, double maxNorm) {
        this.mode = mode;
        this.maxValue = maxValue;
        this.maxNorm = maxNorm;
        this.lastGlobalNorm = 0;
        this.lastScale = 1.0;
    }

    public boolean clip(Collection<Tensor> parameters) {
        if (mode == ClipMode.BY_VALUE) {
            return clipByValue(parameters);
        } else {
            return clipByNorm(parameters);
        }
    }

    private boolean clipByValue(Collection<Tensor> parameters) {
        throw new UnsupportedOperationException(
                "BY_VALUE clipping requires Tensor.clamp() which is not in Aljabr API. " +
                        "Use ClipMode.BY_NORM instead.");
    }

    private boolean clipByNorm(Collection<Tensor> parameters) {
        double globalNormSq = 0.0;
        int gradCount = 0;

        for (Tensor param : parameters) {
            Tensor grad = param.grad();
            if (grad == null)
                continue;

            Tensor squared = grad.mul(grad);
            float sumSq = squared.sum().item();
            globalNormSq += sumSq;
            gradCount++;
        }

        if (gradCount == 0)
            return false;

        double globalNorm = Math.sqrt(globalNormSq);
        this.lastGlobalNorm = globalNorm;

        if (globalNorm <= maxNorm || globalNorm == 0.0) {
            this.lastScale = 1.0;
            return false;
        }

        double scale = maxNorm / globalNorm;
        this.lastScale = scale;

        for (Tensor param : parameters) {
            Tensor grad = param.grad();
            if (grad == null)
                continue;

            Tensor scaledGrad = grad.mul((float) scale);
            param.setGrad(scaledGrad);
        }

        return true;
    }

    public double getLastGlobalNorm() {
        return lastGlobalNorm;
    }

    public double getLastScale() {
        return lastScale;
    }

    public ClipMode getMode() {
        return mode;
    }

    public double getMaxValue() {
        return maxValue;
    }

    public double getMaxNorm() {
        return maxNorm;
    }
}
