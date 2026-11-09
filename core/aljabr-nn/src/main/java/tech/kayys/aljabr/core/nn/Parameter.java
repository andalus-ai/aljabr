package tech.kayys.aljabr.core.nn;

import java.util.Objects;
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

    Parameter(String name, Tensor tensor) {
        this.name = Objects.requireNonNull(name, "name must not be null");
        this.tensor = Objects.requireNonNull(tensor, "tensor must not be null");
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

    void replaceTensor(Tensor replacement) {
        this.tensor = Objects.requireNonNull(replacement, "replacement must not be null");
    }
}
