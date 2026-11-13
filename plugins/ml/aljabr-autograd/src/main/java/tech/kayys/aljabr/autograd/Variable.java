package tech.kayys.aljabr.autograd;

import java.util.List;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * A tensor value participating in an {@link AutogradContext} graph.
 *
 * <p>This is intentionally a distinct wrapper rather than an implementation of
 * {@code Tensor}. It makes differentiation opt-in and avoids a dependency from
 * the foundational tensor module back to the autograd implementation.</p>
 */
public final class Variable {

    @FunctionalInterface
    interface BackwardRule {
        Tensor[] apply(Tensor outputGradient);
    }

    private final AutogradContext context;
    private final Tensor value;
    private final boolean requiresGrad;
    private final List<Variable> parents;
    private final BackwardRule backwardRule;
    private final String operationName;

    Variable(
            AutogradContext context,
            Tensor value,
            boolean requiresGrad,
            List<Variable> parents,
            BackwardRule backwardRule) {
        this(context, value, requiresGrad, parents, backwardRule, "leaf");
    }

    private Variable(
            AutogradContext context,
            Tensor value,
            boolean requiresGrad,
            List<Variable> parents,
            BackwardRule backwardRule,
            String operationName) {
        this.context = Objects.requireNonNull(context, "context must not be null");
        this.value = Objects.requireNonNull(value, "value must not be null");
        this.requiresGrad = requiresGrad;
        this.parents = List.copyOf(parents);
        this.backwardRule = backwardRule;
        this.operationName = operationName;
    }

    /** Returns the borrowed tensor value; its lifetime is managed by its owner/context. */
    public Tensor value() {
        context.ensureOpen();
        return value;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    /** Returns the context-owned accumulated gradient, or {@code null} before backward. */
    public Tensor grad() {
        return context.gradientOf(this);
    }

    public void backward() {
        context.backward(this);
    }

    public Variable add(Variable other) {
        requireCompatibleContext(other);
        if (java.util.Arrays.equals(value.shape().dims(), other.value.shape().dims())) {
            return binary(other, "add", Tensor::add,
                    gradient -> new Tensor[] {
                            requiresGrad ? context.own(gradient.add(0.0f)) : null,
                            other.requiresGrad ? context.own(gradient.add(0.0f)) : null
                    });
        }

        // The first NN adapter needs the common [batch, features] + [features]
        // bias case. Other broadcast patterns remain explicitly unsupported.
        if (value.shape().rank() == 2
                && other.value.shape().rank() == 1
                && value.shape().dim(1) == other.value.shape().dim(0)) {
            Tensor result = context.own(value.add(other.value));
            boolean track = requiresGrad || other.requiresGrad;
            return new Variable(context, result, track, List.of(this, other), gradient -> {
                Tensor leftGradient = requiresGrad ? context.own(gradient.add(0.0f)) : null;
                Tensor rightGradient = other.requiresGrad
                        ? context.own(gradient.sum(0, false))
                        : null;
                return new Tensor[] { leftGradient, rightGradient };
            }, "addRowVector");
        }
        throw new UnsupportedOperationException(
                "Autograd add supports equal shapes and [batch, features] + [features]; other broadcasting is not implemented");
    }

    public Variable sub(Variable other) {
        return binary(other, "subtract", Tensor::sub,
                gradient -> new Tensor[] {
                        requiresGrad ? context.own(gradient.add(0.0f)) : null,
                        other.requiresGrad ? context.own(gradient.mul(-1.0f)) : null
                });
    }

    public Variable mul(Variable other) {
        return binary(other, "multiply", Tensor::mul,
                gradient -> new Tensor[] {
                        requiresGrad ? context.own(gradient.mul(other.value)) : null,
                        other.requiresGrad ? context.own(gradient.mul(value)) : null
                });
    }

    public Variable div(Variable other) {
        return binary(other, "divide", Tensor::div, gradient -> {
            Tensor leftGradient = requiresGrad ? context.own(gradient.div(other.value)) : null;
            Tensor rightGradient = null;
            if (other.requiresGrad) {
                Tensor numerator = context.own(gradient.mul(value));
                Tensor denominatorSquared = context.own(other.value.mul(other.value));
                Tensor quotient = context.own(numerator.div(denominatorSquared));
                rightGradient = context.own(quotient.mul(-1.0f));
            }
            return new Tensor[] { leftGradient, rightGradient };
        });
    }

    public Variable add(float scalar) {
        context.ensureOpen();
        return unary("addScalar", value.add(scalar),
                gradient -> new Tensor[] { requiresGrad ? context.own(gradient.add(0.0f)) : null });
    }

    public Variable mul(float scalar) {
        context.ensureOpen();
        return unary("multiplyScalar", value.mul(scalar),
                gradient -> new Tensor[] { requiresGrad ? context.own(gradient.mul(scalar)) : null });
    }

    public Variable div(float scalar) {
        context.ensureOpen();
        if (scalar == 0.0f) {
            throw new IllegalArgumentException("Scalar divisor must not be zero");
        }
        return unary("divideScalar", value.div(scalar),
                gradient -> new Tensor[] { requiresGrad ? context.own(gradient.div(scalar)) : null });
    }

    /** Matrix multiplication for rank-two tensors only. */
    public Variable matmul(Variable other) {
        requireCompatibleContext(other);
        if (value.shape().rank() != 2 || other.value.shape().rank() != 2) {
            throw new UnsupportedOperationException("Autograd matmul currently supports rank-two tensors only");
        }
        boolean track = requiresGrad || other.requiresGrad;
        Tensor result = context.own(value.matmul(other.value));
        return new Variable(context, result, track, List.of(this, other), gradient -> {
            Tensor leftGradient = null;
            Tensor rightGradient = null;
            if (requiresGrad) {
                Tensor rightTranspose = context.own(other.value.transpose(0, 1));
                leftGradient = context.own(gradient.matmul(rightTranspose));
            }
            if (other.requiresGrad) {
                Tensor leftTranspose = context.own(value.transpose(0, 1));
                rightGradient = context.own(leftTranspose.matmul(gradient));
            }
            return new Tensor[] { leftGradient, rightGradient };
        }, "matmul");
    }


    /** Rank-two transpose with the inverse transpose as its vector-Jacobian product. */
    public Variable transpose(int dim0, int dim1) {
        context.ensureOpen();
        if (value.shape().rank() != 2) {
            throw new UnsupportedOperationException("Autograd transpose currently supports rank-two tensors only");
        }
        Tensor result = value.transpose(dim0, dim1);
        return unaryWithResult("transpose", result,
                gradient -> new Tensor[] { requiresGrad ? context.own(gradient.transpose(dim0, dim1)) : null });
    }

    /**
     * Adds a rank-one feature bias to each row of a rank-two value.
     * The bias gradient is reduced over the batch dimension.
     */
    public Variable addBias(Variable bias) {
        requireCompatibleContext(bias);
        if (value.shape().rank() != 2 || bias.value.shape().rank() != 1
                || value.shape().dim(1) != bias.value.shape().dim(0)) {
            throw new UnsupportedOperationException(
                    "Autograd addBias requires [batch, features] plus [features]");
        }
        Tensor result = context.own(value.add(bias.value));
        return new Variable(context, result, requiresGrad || bias.requiresGrad,
                List.of(this, bias), gradient -> new Tensor[] {
                    requiresGrad ? context.own(gradient.add(0.0f)) : null,
                    bias.requiresGrad ? context.own(gradient.sum(0, false)) : null
                }, "addBias");
    }

    private Variable unaryWithResult(String name, Tensor result, BackwardRule rule) {
        context.ensureOpen();
        return new Variable(context, context.own(result), requiresGrad, List.of(this), rule, name);
    }

    public Variable sum() {
        context.ensureOpen();
        return unary("sum", value.sum(), gradient -> {
            if (!requiresGrad) {
                return new Tensor[] { null };
            }
            Tensor ones = context.own(Tensor.ones(value.backend(), value.shape().dims()));
            return new Tensor[] { context.own(ones.mul(gradient.item())) };
        });
    }

    public Variable mean() {
        context.ensureOpen();
        return unary("mean", value.mean(), gradient -> {
            if (!requiresGrad) {
                return new Tensor[] { null };
            }
            Tensor ones = context.own(Tensor.ones(value.backend(), value.shape().dims()));
            return new Tensor[] { context.own(ones.mul(gradient.item() / value.numel())) };
        });
    }

    private Variable unary(String name, Tensor result, BackwardRule rule) {
        context.ensureOpen();
        boolean track = requiresGrad;
        Tensor owned = context.own(result);
        return new Variable(context, owned, track, List.of(this), rule, name);
    }

    private Variable binary(
            Variable other,
            String name,
            TensorBinary forward,
            BackwardRule rule) {
        requireCompatibleContext(other);
        if (!java.util.Arrays.equals(value.shape().dims(), other.value.shape().dims())) {
            throw new UnsupportedOperationException(
                    "Autograd " + name + " currently requires identical input shapes; broadcasting is not implemented");
        }
        Tensor result = context.own(forward.apply(value, other.value));
        boolean track = requiresGrad || other.requiresGrad;
        return new Variable(context, result, track, List.of(this, other), rule, name);
    }

    private void requireCompatibleContext(Variable other) {
        context.ensureOpen();
        Objects.requireNonNull(other, "other must not be null");
        other.context.ensureOpen();
        if (other.context != context) {
            throw new IllegalArgumentException("Variables must belong to the same autograd context");
        }
        if (value.backend() != other.value.backend()) {
            throw new IllegalArgumentException("Variables must use the same compute backend instance");
        }
    }

    AutogradContext context() {
        return context;
    }

    List<Variable> parents() {
        return parents;
    }

    BackwardRule backwardRule() {
        return backwardRule;
    }

    String operationName() {
        return operationName;
    }

    @FunctionalInterface
    private interface TensorBinary {
        Tensor apply(Tensor left, Tensor right);
    }
}
