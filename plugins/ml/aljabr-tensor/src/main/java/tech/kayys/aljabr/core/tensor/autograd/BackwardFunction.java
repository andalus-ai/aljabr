package tech.kayys.aljabr.core.tensor.autograd;

import java.util.List;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Computes the gradients contributed to an operation's inputs during reverse-mode
 * differentiation.
 *
 * <p>The returned array is positional: entry {@code i} is the gradient for input
 * {@code i}. An entry may be {@code null} when that input is non-differentiable
 * or does not require a gradient. The array length must equal the number of
 * inputs supplied to the function.</p>
 *
 * <p>Each non-null returned tensor is a newly owned result. It must not alias
 * the supplied output, output gradient, an input, or another returned entry.
 * The caller is responsible for releasing returned tensors when they are no
 * longer needed. Arguments and tensors supplied to this function are borrowed
 * and remain owned by the caller.</p>
 *
 * <p>This contract is deliberately expressed only in terms of {@link Tensor};
 * implementations can use the tensor's backend without coupling the graph to a
 * particular device, storage representation, or IR.</p>
 */
@FunctionalInterface
public interface BackwardFunction {

    /**
     * Applies the local vector-Jacobian product for one recorded operation.
     *
     * @param output materialized forward output (borrowed)
     * @param outputGradient gradient propagated from consumers (borrowed)
     * @param inputs materialized operation inputs, in forward argument order (borrowed)
     * @param arguments operation-specific forward arguments (borrowed)
     * @return one newly owned gradient per input, with {@code null} for inputs that receive no gradient
     */
    Tensor[] apply(Tensor output, Tensor outputGradient, List<Tensor> inputs, Object[] arguments);
}
