package tech.kayys.aljabr.optim;

import java.util.Map;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Applies gradient updates to registered model parameters.
 *
 * <p>This module depends on the NN parameter contract, not on eager autograd.
 * Callers provide gradients produced by any compatible differentiation system.
 * Optimizer implementations own their algorithm state and replace parameter
 * tensor values through {@link Parameter#updateTensor(Tensor)}.</p>
 */
public interface Optimizer {

    /**
     * Applies one update for each trainable parameter with a supplied gradient.
     * Parameter names should be stable and fully qualified within the model.
     */
    void step(Map<String, Parameter> parameters, Map<String, Tensor> gradients);

    /** Clears accumulated optimizer state, retaining hyperparameters. */
    void reset();
}
