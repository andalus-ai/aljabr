package tech.kayys.aljabr.autograd.nn;

import java.util.Objects;

import tech.kayys.aljabr.autograd.AutogradContext;
import tech.kayys.aljabr.autograd.Variable;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;

/**
 * Explicit differentiable adapter for an existing {@link Linear} module.
 *
 * <p>The wrapped module remains unchanged and usable for ordinary inference.
 * This adapter records the linear operation in a caller-owned autograd context.
 * Parameter variables are snapshots of the current registered parameter tensors
 * and must not be used after the context is closed or after parameters are
 * replaced.</p>
 */
public final class DifferentiableLinear {

    private final Linear layer;
    private final AutogradContext context;
    private final Variable weight;
    private final Variable bias;

    public DifferentiableLinear(Linear layer, AutogradContext context) {
        this.layer = Objects.requireNonNull(layer, "layer must not be null");
        this.context = Objects.requireNonNull(context, "context must not be null");
        Parameter weightParameter = Objects.requireNonNull(
                layer.namedParameterObjects().get("weight"), "Linear weight parameter is missing");
        this.weight = context.variable(weightParameter.tensor(), weightParameter.requiresGrad());
        Parameter biasParameter = layer.namedParameterObjects().get("bias");
        this.bias = biasParameter == null
                ? null
                : context.variable(biasParameter.tensor(), biasParameter.requiresGrad());
    }

    /** Records y = x W^T + b using the existing module's live parameter tensors. */
    public Variable forward(Variable input) {
        Objects.requireNonNull(input, "input must not be null");
        if (input.value().backend() != weight.value().backend()) {
            throw new IllegalArgumentException("Input and Linear parameters must use the same backend instance");
        }
        Variable output = input.matmul(weight.transpose(0, 1));
        return bias == null ? output : output.addBias(bias);
    }

    /** Differentiable handle for the module's weight parameter. */
    public Variable weight() {
        return weight;
    }

    /** Differentiable handle for the module's bias parameter, or {@code null}. */
    public Variable bias() {
        return bias;
    }

    public Linear layer() {
        return layer;
    }
}
