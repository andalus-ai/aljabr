package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * 2D Batch Normalization layer.
 * @author bhangun
 */
public class BatchNorm2d extends NNModule {

    private final long numFeatures;
    private final float eps;
    private final float momentum;
    private final boolean affine;
    private final boolean trackRunningStats;

    public BatchNorm2d(long numFeatures, float eps, float momentum,
            boolean affine, boolean trackRunningStats) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        if (affine) {
            registerParameter("weight", Tensor.ones(numFeatures));
            registerParameter("bias", Tensor.zeros(numFeatures));
        }

        if (trackRunningStats) {
            registerBuffer("running_mean", Tensor.zeros(numFeatures));
            registerBuffer("running_var", Tensor.ones(numFeatures));
        }
    }

    public BatchNorm2d(long numFeatures) {
        this(numFeatures, 1e-5f, 0.1f, true, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = affine ? parameters.get("weight") : null;
        Tensor bias = affine ? parameters.get("bias") : null;
        Tensor runningMean = trackRunningStats ? buffers.get("running_mean") : null;
        Tensor runningVar = trackRunningStats ? buffers.get("running_var") : null;

        return input.batchNorm(weight, bias, runningMean, runningVar,
                training, momentum, eps);
    }

    public long numFeatures() {
        return numFeatures;
    }

    public float eps() {
        return eps;
    }

    public float momentum() {
        return momentum;
    }

    public boolean isAffine() {
        return affine;
    }

    public boolean isTrackRunningStats() {
        return trackRunningStats;
    }

    @Override
    public String toString() {
        return String.format("BatchNorm2d(%d, eps=%.1e, momentum=%.2f, affine=%b)",
                numFeatures, eps, momentum, affine);
    }
}
