package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.Collection;
import java.util.Map;

/**
 * Optimizer contract for parameter updates during training.
 * @author bhangun
 */
public interface Optimizer {

    /**
     * Perform an optimization step on a collection of tensors.
     * This is used by the Trainer for standard neural network training.
     * 
     * @param params collection of parameter tensors to update in-place.
     */
    void step(Collection<Tensor> params);

    /**
     * Perform an optimization step using named parameters and gradients.
     * This is useful for more complex training scenarios or IR-based training.
     * 
     * @param params mutable map of parameter name -> current parameter tensor.
     * @param grads  map of parameter name -> gradient tensor.
     */
    default void step(Map<String, Tensor> params, Map<String, Tensor> grads) {
        // Optional implementation
    }
}