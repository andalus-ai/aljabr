package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Layer Normalization.
 * @author bhangun
 */
public class LayerNorm extends NNModule {

    private final long[] normalizedShape;
    private final float eps;
    private final boolean elementwiseAffine;

    public LayerNorm(long[] normalizedShape, float eps, boolean elementwiseAffine) {
        this.normalizedShape = normalizedShape;
        this.eps = eps;
        this.elementwiseAffine = elementwiseAffine;

        if (elementwiseAffine) {
            registerParameter("weight", Tensor.ones(normalizedShape));
            registerParameter("bias", Tensor.zeros(normalizedShape));
        }
    }

    public LayerNorm(long... normalizedShape) {
        this(normalizedShape, 1e-5f, true);
    }

    public LayerNorm(Tensor weight, Tensor bias, float eps) {
        this.normalizedShape = new long[weight.shape().rank()];
        for (int i = 0; i < this.normalizedShape.length; i++) {
            this.normalizedShape[i] = weight.shape().dim(i);
        }
        this.eps = eps;
        this.elementwiseAffine = true;
        registerParameter("weight", weight);
        if (bias != null) {
            registerParameter("bias", bias);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = elementwiseAffine ? parameters.get("weight") : null;
        Tensor bias = elementwiseAffine ? parameters.get("bias") : null;
        return input.layerNorm(normalizedShape, weight, bias, eps);
    }

    public long[] normalizedShape() {
        return normalizedShape;
    }

    public float eps() {
        return eps;
    }

    public boolean isElementwiseAffine() {
        return elementwiseAffine;
    }

    @Override
    public String toString() {
        return String.format("LayerNorm(%s, eps=%.1e)", java.util.Arrays.toString(normalizedShape), eps);
    }
}
