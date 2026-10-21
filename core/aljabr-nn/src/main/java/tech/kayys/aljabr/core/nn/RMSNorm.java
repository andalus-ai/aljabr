package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class RMSNorm extends NNModule {
    private final int dim;
    private final float eps;

    public RMSNorm(int dim, float eps) {
        this.dim = dim;
        this.eps = eps;

        // Initialize weight to ones
        Tensor weight = TensorFactory.ones(dim);
        registerParameter("weight", weight);
    }

    // Constructor used when loading pre-trained weights
    public RMSNorm(Tensor weight, float eps) {
        this.dim = (int) weight.shape().dim(0);
        this.eps = eps;
        registerParameter("weight", weight);
    }

    @Override
    public Tensor forward(Tensor input) {
        // RMSNorm(x) = (x / RMS(x)) * weight
        // RMS(x) = sqrt(mean(x^2) + eps)
        Tensor weight = parameters.get("weight");

        // square -> mean over last dim -> add eps -> sqrt
        Tensor x2 = input.mul(input);
        Tensor mean_x2 = x2.mean(-1, true);
        Tensor rms = mean_x2.add(eps).pow(0.5f);

        Tensor normalized = input.div(rms);
        return normalized.mul(weight);
    }
}
