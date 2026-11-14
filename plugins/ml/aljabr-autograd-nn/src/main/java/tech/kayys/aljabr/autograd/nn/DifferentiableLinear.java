package tech.kayys.aljabr.autograd.nn;

import java.util.Objects;

import tech.kayys.aljabr.autograd.AutogradContext;
import tech.kayys.aljabr.autograd.Variable;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Explicit differentiable adapter for an existing {@link Linear} module.
 *
 * <p>The wrapped module remains unchanged and usable for ordinary inference.
 * This adapter binds the parameter tensors present at construction time into
 * the supplied autograd context. Those tensors are borrowed by the context:
 * the module and its parameters must remain alive and must not be replaced
 * while this adapter is in use.</p>
 *
 * <p>Parameter gradients are exposed as context-owned tensors. This adapter
 * does not apply updates or own optimizer state.</p>
 */
public final class DifferentiableLinear {

    private final Linear layer;
    private final AutogradContext context;
    private final Parameter weightParameter;
    private final Parameter biasParameter;
    private final Variable weight;
    private final Variable bias;

    public DifferentiableLinear(Linear layer, AutogradContext context) {
        this.layer = Objects.requireNonNull(layer, "layer must not be null");
        this.context = Objects.requireNonNull(context, "context must not be null");
        this.weightParameter = Objects.requireNonNull(
                layer.namedParameterObjects().get("weight"), "Linear weight parameter is missing");
        this.weight = context.variable(weightParameter.tensor(), weightParameter.requiresGrad());
        this.biasParameter = layer.namedParameterObjects().get("bias");
        this.bias = biasParameter == null
                ? null
                : context.variable(biasParameter.tensor(), biasParameter.requiresGrad());
    }

    /** Records y = x W^T + b using the parameter tensors bound at construction. */
    public Variable forward(Variable input) {
        Objects.requireNonNull(input, "input must not be null");
        ensureParameterSnapshotIsCurrent();
        if (input.value().backend() != weight.value().backend()) {
            throw new IllegalArgumentException("Input and Linear parameters must use the same backend instance");
        }
        Variable output = input.matmul(weight.transpose(0, 1));
        return bias == null ? output : output.addBias(bias);
    }

    /** Differentiable handle for the module's weight parameter. */
    public Variable weight() {
        ensureParameterSnapshotIsCurrent();
        return weight;
    }

    /** Differentiable handle for the module's bias parameter, or {@code null}. */
    public Variable bias() {
        ensureParameterSnapshotIsCurrent();
        return bias;
    }

    /** Registered weight handle; no tensor ownership is transferred. */
    public Parameter weightParameter() {
        ensureParameterSnapshotIsCurrent();
        return weightParameter;
    }

    /** Registered bias handle, or {@code null} when the layer has no bias. */
    public Parameter biasParameter() {
        ensureParameterSnapshotIsCurrent();
        return biasParameter;
    }

    /**
     * Returns the context-owned weight gradient after a successful backward
     * pass, or {@code null} before one. The returned tensor is borrowed and
     * remains valid only until the autograd context is closed.
     */
    public Tensor weightGradient() {
        ensureParameterSnapshotIsCurrent();
        Tensor gradient = weight.grad();
        if (gradient != null) {
            weightParameter.validateGradient(gradient);
        }
        return gradient;
    }

    /** Bias-gradient counterpart to {@link #weightGradient()}; null without bias or before backward. */
    public Tensor biasGradient() {
        ensureParameterSnapshotIsCurrent();
        if (bias == null) {
            return null;
        }
        Tensor gradient = bias.grad();
        if (gradient != null) {
            biasParameter.validateGradient(gradient);
        }
        return gradient;
    }

    public Linear layer() {
        return layer;
    }

    private void ensureParameterSnapshotIsCurrent() {
        context.ensureOpen();
        if (weightParameter.tensor() != weight.value()) {
            throw new IllegalStateException(
                    "Linear weight parameter was replaced after the differentiable adapter was created");
        }
        if (biasParameter != null && biasParameter.tensor() != bias.value()) {
            throw new IllegalStateException(
                    "Linear bias parameter was replaced after the differentiable adapter was created");
        }
    }
}
