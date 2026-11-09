package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Root Mean Square normalization layer.
 */
public class RMSNorm extends NNModule {
    private final int dim;
    private final float eps;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public RMSNorm(int dim, float eps) {
        this(null, dim, eps);
    }

    public RMSNorm(ComputeBackend backend, int dim, float eps) {
        super(backend);
        this.dim = dim;
        this.eps = eps;
        registerParameter("weight", ones(dim));
    }

    public RMSNorm(Tensor weight, float eps) {
        super(Objects.requireNonNull(weight, "weight must not be null").backend());
        this.dim = Math.toIntExact(weight.shape().dim(0));
        this.eps = eps;
        registerParameter("weight", weight);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        Tensor weight = parameters.get("weight");
        Tensor x2 = input.mul(input);
        Tensor meanX2 = x2.mean(-1, true);
        Tensor rms = meanX2.add(eps).pow(0.5f);
        Tensor normalized = input.div(rms);
        return normalized.mul(weight);
    }

    public int dim() { return dim; }
    public float eps() { return eps; }
}
