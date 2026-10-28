package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.Collection;
import java.util.Map;

/**
 * Optimizer contract for parameter updates during training.
 */
public interface Optimizer {

    /**
     * Perform an optimization step on a collection of tensors.
     *
     * <p>This overload is only supported by implementations that can mutate
     * parameter storage. Aljabr's current {@link Tensor} arithmetic returns
     * new tensors and has no assignment operation, so implementations may
     * throw {@link UnsupportedOperationException}. Prefer the named-map
     * overload until a mutable parameter abstraction is available.
     *
     * @param params collection of parameter tensors to update
     */
    void step(Collection<Tensor> params);

    /**
     * Perform an optimization step using named parameters and gradients.
     * Implementations may replace map values with updated tensors.
     *
     * @param params mutable map of parameter name to current parameter tensor
     * @param grads map of parameter name to gradient tensor
     */
    default void step(Map<String, Tensor> params, Map<String, Tensor> grads) {
        throw new UnsupportedOperationException("Named-parameter updates are not implemented by this optimizer");
    }
}
