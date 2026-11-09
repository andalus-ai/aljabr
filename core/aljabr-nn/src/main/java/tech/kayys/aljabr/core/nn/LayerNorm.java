package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Layer Normalization.
 */
public class LayerNorm extends NNModule {

    private final long[] normalizedShape;
    private final float eps;
    private final boolean elementwiseAffine;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public LayerNorm(long[] normalizedShape, float eps, boolean elementwiseAffine) {
        this(null, normalizedShape, eps, elementwiseAffine);
    }

    public LayerNorm(ComputeBackend backend, long[] normalizedShape, float eps, boolean elementwiseAffine) {
        super(backend);
        this.normalizedShape = Objects.requireNonNull(normalizedShape, "normalizedShape must not be null").clone();
        this.eps = eps;
        this.elementwiseAffine = elementwiseAffine;
        if (elementwiseAffine) {
            registerParameter("weight", ones(this.normalizedShape));
            registerParameter("bias", zeros(this.normalizedShape));
        }
    }

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public LayerNorm(long... normalizedShape) {
        this(null, normalizedShape, 1e-5f, true);
    }

    public LayerNorm(ComputeBackend backend, long... normalizedShape) {
        this(backend, normalizedShape, 1e-5f, true);
    }

    public LayerNorm(Tensor weight, Tensor bias, float eps) {
        super(Objects.requireNonNull(weight, "weight must not be null").backend());
        this.normalizedShape = weight.shape().dims();
        this.eps = eps;
        this.elementwiseAffine = true;
        registerParameter("weight", weight);
        if (bias != null) {
            registerParameter("bias", bias);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        Tensor weight = elementwiseAffine ? parameters.get("weight") : null;
        Tensor bias = elementwiseAffine ? parameters.get("bias") : null;
        return input.layerNorm(normalizedShape, weight, bias, eps);
    }

    public long[] normalizedShape() { return normalizedShape.clone(); }
    public float eps() { return eps; }
    public boolean isElementwiseAffine() { return elementwiseAffine; }

    @Override
    public String toString() {
        return String.format("LayerNorm(%s, eps=%.1e)", java.util.Arrays.toString(normalizedShape), eps);
    }
}
