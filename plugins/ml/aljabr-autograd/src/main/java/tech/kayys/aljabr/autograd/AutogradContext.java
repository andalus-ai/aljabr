package tech.kayys.aljabr.autograd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Owns one eager differentiation graph and all tensors created by its recorded
 * operations and backward pass.
 *
 * <p>Leaf tensors are borrowed: the caller must keep them alive until backward
 * has completed. Intermediate values and gradients are owned by this context
 * and remain valid until {@link #close()}.</p>
 *
 * <p>This initial implementation is deliberately single-use and thread-confined.
 * It supports scalar and explicitly seeded reverse-mode differentiation without broadcasting.</p>
 */
public final class AutogradContext implements AutoCloseable {

    private final List<Tensor> ownedTensors = new ArrayList<>();
    private final Map<Variable, Tensor> gradients = new IdentityHashMap<>();
    private boolean backwardCompleted;
    private boolean closed;

    /**
     * Tracks an existing eager tensor without taking ownership of it.
     *
     * @param value caller-owned leaf value
     * @param requiresGrad whether this leaf participates in differentiation
     * @return a variable bound to this context
     */
    public Variable variable(Tensor value, boolean requiresGrad) {
        ensureOpen();
        Objects.requireNonNull(value, "value must not be null");
        if (value.backend() == null) {
            throw new IllegalArgumentException("Autograd requires a tensor with an explicit backend");
        }
        if (value.dtype() != tech.kayys.aljabr.core.tensor.DType.F32) {
            throw new IllegalArgumentException("Initial autograd prototype supports F32 tensors only");
        }
        return new Variable(this, value, requiresGrad, List.of(), null);
    }

    /** Tracks a caller-owned tensor as a differentiable leaf. */
    public Variable variable(Tensor value) {
        return variable(value, true);
    }

    Tensor own(Tensor tensor) {
        ensureOpen();
        ownedTensors.add(Objects.requireNonNull(tensor, "owned tensor must not be null"));
        return tensor;
    }

    Tensor gradientOf(Variable variable) {
        ensureOpen();
        return gradients.get(variable);
    }

    void backward(Variable output) {
        validateOutput(output);
        if (!output.requiresGrad()) {
            return;
        }
        if (output.value().numel() != 1) {
            throw new IllegalArgumentException(
                    "Implicit backward requires a scalar output; use backward(Tensor outputGradient) for non-scalar outputs");
        }
        Tensor seed = own(TensorFactory.ones(output.value().backend(), output.value().shape().dims()));
        runBackward(output, seed);
    }

    void backward(Variable output, Tensor outputGradient) {
        validateOutput(output);
        Objects.requireNonNull(outputGradient, "outputGradient must not be null");
        if (!output.requiresGrad()) {
            return;
        }
        if (outputGradient.backend() != output.value().backend()) {
            throw new IllegalArgumentException("Output gradient must use the same backend instance as the output");
        }
        if (outputGradient.dtype() != tech.kayys.aljabr.core.tensor.DType.F32) {
            throw new IllegalArgumentException("Initial autograd prototype supports F32 output gradients only");
        }
        if (!java.util.Arrays.equals(outputGradient.shape().dims(), output.value().shape().dims())) {
            throw new IllegalArgumentException("Output gradient shape mismatch: expected "
                    + output.value().shape() + " but got " + outputGradient.shape());
        }

        // Copy the caller's seed so the context owns a stable value for the full backward pass.
        Tensor seed = own(outputGradient.add(0.0f));
        runBackward(output, seed);
    }

    private void validateOutput(Variable output) {
        ensureOpen();
        Objects.requireNonNull(output, "output must not be null");
        if (output.context() != this) {
            throw new IllegalArgumentException("Output belongs to a different autograd context");
        }
        if (backwardCompleted) {
            throw new IllegalStateException("This autograd context has already run backward");
        }
    }

    private void runBackward(Variable output, Tensor seed) {
        backwardCompleted = true;
        gradients.put(output, seed);

        List<Variable> order = new ArrayList<>();
        Set<Variable> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        visit(output, visited, order);

        for (int index = order.size() - 1; index >= 0; index--) {
            Variable current = order.get(index);
            Tensor outputGradient = gradients.get(current);
            if (outputGradient == null || !current.requiresGrad()) {
                continue;
            }
            if (current.backwardRule() == null) {
                if (!current.parents().isEmpty()) {
                    throw new UnsupportedOperationException(
                            "No backward rule registered for operation " + current.operationName());
                }
                continue;
            }

            Tensor[] contributions = current.backwardRule().apply(outputGradient);
            if (contributions.length != current.parents().size()) {
                for (Tensor contribution : contributions) {
                    if (contribution != null) {
                        own(contribution);
                    }
                }
                throw new IllegalStateException("Backward rule for " + current.operationName()
                        + " returned " + contributions.length + " gradients for "
                        + current.parents().size() + " inputs");
            }
            for (int i = 0; i < contributions.length; i++) {
                Tensor contribution = contributions[i];
                if (contribution == null) {
                    continue;
                }
                own(contribution);
                Variable parent = current.parents().get(i);
                if (!parent.requiresGrad()) {
                    throw new IllegalStateException("Backward rule produced a gradient for a non-trainable input");
                }
                if (!java.util.Arrays.equals(contribution.shape().dims(), parent.value().shape().dims())) {
                    throw new IllegalStateException("Gradient shape mismatch for " + current.operationName()
                            + ": expected " + parent.value().shape() + " but got " + contribution.shape());
                }
                Tensor previous = gradients.get(parent);
                gradients.put(parent, previous == null ? contribution : own(previous.add(contribution)));
            }
        }
    }

    private static void visit(Variable variable, Set<Variable> visited, List<Variable> order) {
        if (!visited.add(variable)) {
            return;
        }
        for (Variable parent : variable.parents()) {
            visit(parent, visited, order);
        }
        order.add(variable);
    }

    void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Autograd context has been closed");
        }
    }

    /**
     * Releases intermediate values and gradients. Leaf tensors remain owned by
     * their callers and are not released.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            for (int i = ownedTensors.size() - 1; i >= 0; i--) {
                ownedTensors.get(i).release();
            }
            ownedTensors.clear();
            gradients.clear();
        }
    }
}
