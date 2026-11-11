package tech.kayys.aljabr.core.nn;

import java.util.Objects;
import java.util.function.Consumer;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Model-level handle for a tensor registered as a neural-network parameter.
 *
 * <p>The parameter does not own separate storage: it delegates gradient
 * semantics to its tensor and exposes the same live tensor instance used by
 * module execution. The module controls tensor replacement when parameters
 * are moved between devices.</p>
 */
public final class Parameter {

    private final String name;
    private volatile Tensor tensor;
    private final Consumer<Tensor> updateListener;

    Parameter(String name, Tensor tensor, Consumer<Tensor> updateListener) {
        this.name = Objects.requireNonNull(name, "name must not be null");
        this.tensor = Objects.requireNonNull(tensor, "tensor must not be null");
        this.updateListener = Objects.requireNonNull(updateListener, "updateListener must not be null");
    }

    /** Local registration name within the owning module. */
    public String name() {
        return name;
    }

    /** Returns the live tensor backing this parameter. */
    public Tensor tensor() {
        return tensor;
    }

    /** Whether autograd should track operations involving this parameter. */
    public boolean requiresGrad() {
        return tensor.requiresGrad();
    }

    /** Enables or disables gradient tracking on the underlying tensor. */
    public void setRequiresGrad(boolean requiresGrad) {
        tensor.setRequiresGrad(requiresGrad);
    }

    /** Convenience predicate for optimizer parameter filtering. */
    public boolean isTrainable() {
        return requiresGrad();
    }

    /**
     * Replaces this parameter's value while preserving its registration and
     * trainability state.
     *
     * <p>The replacement must retain the parameter's shape, dtype, backend,
     * and device. This is intended for optimizers that produce a new tensor
     * value because tensor arithmetic is immutable.</p>
     *
     * <p>The previous tensor is not released: callers may still hold references
     * to it (for example, from an in-flight forward pass). Resource ownership
     * remains with the module/runtime lifecycle.</p>
     *
     * @param replacement updated tensor value
     */
    public void updateTensor(Tensor replacement) {
        Objects.requireNonNull(replacement, "replacement must not be null");
        Tensor current = tensor;
        if (!java.util.Arrays.equals(current.shape().dims(), replacement.shape().dims())) {
            throw new IllegalArgumentException("Parameter shape cannot change from "
                    + current.shape() + " to " + replacement.shape());
        }
        if (current.dtype() != replacement.dtype()) {
            throw new IllegalArgumentException("Parameter dtype cannot change from "
                    + current.dtype() + " to " + replacement.dtype());
        }
        if (current.backend() != replacement.backend()) {
            throw new IllegalArgumentException("Parameter backend cannot change during an optimizer update");
        }
        if (current.device() != replacement.device()) {
            throw new IllegalArgumentException("Parameter device cannot change during an optimizer update");
        }
        replacement.setRequiresGrad(current.requiresGrad());
        this.tensor = replacement;
        updateListener.accept(replacement);
    }

    void replaceTensor(Tensor replacement) {
        this.tensor = Objects.requireNonNull(replacement, "replacement must not be null");
    }
}
