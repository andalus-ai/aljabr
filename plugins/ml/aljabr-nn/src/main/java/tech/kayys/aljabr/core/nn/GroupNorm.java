package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Group normalization over the channel and spatial dimensions of each sample.
 * Inputs use the {@code [N, C, ...]} layout.
 */
public class GroupNorm extends NNModule {

    private static final float DEFAULT_EPS = 1e-5f;

    private final int numGroups;
    private final long numChannels;
    private final float eps;
    private final boolean affine;

    /** @deprecated Use a constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public GroupNorm(int numGroups, long numChannels) {
        this(null, numGroups, numChannels, DEFAULT_EPS, true);
    }

    public GroupNorm(ComputeBackend backend, int numGroups, long numChannels) {
        this(backend, numGroups, numChannels, DEFAULT_EPS, true);
    }

    public GroupNorm(ComputeBackend backend, int numGroups, long numChannels, float eps, boolean affine) {
        super(backend);
        if (numGroups <= 0) {
            throw new IllegalArgumentException("numGroups must be positive");
        }
        if (numChannels <= 0) {
            throw new IllegalArgumentException("numChannels must be positive");
        }
        if (numChannels % numGroups != 0) {
            throw new IllegalArgumentException(
                    "numChannels must be divisible by numGroups: " + numChannels + " % " + numGroups);
        }
        if (!(eps > 0.0f) || !Float.isFinite(eps)) {
            throw new IllegalArgumentException("eps must be finite and positive");
        }
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.eps = eps;
        this.affine = affine;
        if (affine) {
            registerParameter("weight", ones(numChannels));
            registerParameter("bias", zeros(numChannels));
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        validateInputBackend(input);

        long[] inputShape = input.shape().dims();
        if (inputShape.length < 2) {
            throw new IllegalArgumentException("GroupNorm expects input rank >= 2: " + input.shape());
        }
        if (inputShape[1] != numChannels) {
            throw new IllegalArgumentException(
                    "GroupNorm expected " + numChannels + " channels but got " + inputShape[1]);
        }

        long elementsPerSample = 1;
        for (int dimension = 1; dimension < inputShape.length; dimension++) {
            elementsPerSample = Math.multiplyExact(elementsPerSample, inputShape[dimension]);
        }
        long groupElements = elementsPerSample / numGroups;

        Tensor grouped = null, mean = null, centered = null, squared = null;
        Tensor variance = null, stabilizedVariance = null, inverseStd = null;
        Tensor normalized = null, reshaped = null, weightView = null, biasView = null, scaled = null;
        try {
            grouped = input.reshape(inputShape[0], numGroups, groupElements);
            mean = grouped.mean(2, true);
            centered = grouped.sub(mean);
            squared = centered.mul(centered);
            variance = squared.mean(2, true);
            stabilizedVariance = variance.add(eps);
            inverseStd = stabilizedVariance.pow(-0.5f);
            normalized = centered.mul(inverseStd);
            reshaped = normalized.reshape(inputShape);

            if (!affine) {
                Tensor output = reshaped;
                reshaped = null;
                return output;
            }

            long[] affineShape = new long[inputShape.length];
            affineShape[0] = 1;
            affineShape[1] = numChannels;
            for (int i = 2; i < affineShape.length; i++) {
                affineShape[i] = 1;
            }
            weightView = parameters.get("weight").reshape(affineShape);
            biasView = parameters.get("bias").reshape(affineShape);
            scaled = reshaped.mul(weightView);
            return scaled.add(biasView);
        } finally {
            release(grouped);
            release(mean);
            release(centered);
            release(squared);
            release(variance);
            release(stabilizedVariance);
            release(inverseStd);
            release(normalized);
            release(reshaped);
            release(weightView);
            release(biasView);
            release(scaled);
        }
    }

    private static void release(Tensor tensor) {
        if (tensor != null) {
            tensor.release();
        }
    }

    public int numGroups() { return numGroups; }
    public long numChannels() { return numChannels; }
    public float eps() { return eps; }
    public boolean isAffine() { return affine; }
}
