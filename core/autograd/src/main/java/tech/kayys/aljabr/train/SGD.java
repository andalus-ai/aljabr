package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;

/**
 * Stochastic Gradient Descent optimizer.
 *
 * <p>The collection-based API cannot update parameters because {@link Tensor}
 * exposes immutable arithmetic and no assignment operation. Use the named-map
 * overload, which replaces entries with updated tensor values, until the
 * tensor API provides an explicit mutable parameter abstraction.
 */
public final class SGD implements Optimizer {
    private final float lr;

    public SGD(float lr) {
        if (!Float.isFinite(lr) || lr < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        this.lr = lr;
    }

    @Override
    public void step(Collection<Tensor> params) {
        Objects.requireNonNull(params, "params");
        throw new UnsupportedOperationException(
                "SGD.step(Collection<Tensor>) cannot replace immutable tensor values; "
                        + "use step(Map<String, Tensor>, Map<String, Tensor>)");
    }

    @Override
    public void step(Map<String, Tensor> params, Map<String, Tensor> grads) {
        Objects.requireNonNull(params, "params");
        Objects.requireNonNull(grads, "grads");

        for (Map.Entry<String, Tensor> entry : params.entrySet()) {
            String name = entry.getKey();
            Tensor parameter = Objects.requireNonNull(entry.getValue(), "parameter " + name);
            Tensor gradient = grads.get(name);
            if (gradient == null || !parameter.requiresGrad()) {
                continue;
            }
            params.put(name, parameter.sub(gradient.mul(lr)));
        }
    }
}
