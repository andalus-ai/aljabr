package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;

/**
 * AdamW optimizer with decoupled weight decay.
 *
 * <p>The collection-based API cannot update parameters because {@link Tensor}
 * exposes immutable arithmetic and no assignment operation. Use the named-map
 * overload, which replaces entries with updated tensor values, until the
 * tensor API provides an explicit mutable parameter abstraction.
 */
public final class AdamW implements Optimizer {
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final OptimizerState state = new OptimizerState();

    public AdamW(float lr, float beta1, float beta2, float eps, float weightDecay) {
        if (!Float.isFinite(lr) || lr < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        if (!Float.isFinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f) {
            throw new IllegalArgumentException("beta1 must be in [0, 1)");
        }
        if (!Float.isFinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f) {
            throw new IllegalArgumentException("beta2 must be in [0, 1)");
        }
        if (!Float.isFinite(eps) || eps <= 0.0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        if (!Float.isFinite(weightDecay) || weightDecay < 0.0f) {
            throw new IllegalArgumentException("weight decay must be finite and non-negative");
        }
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step(Collection<Tensor> params) {
        Objects.requireNonNull(params, "params");
        throw new UnsupportedOperationException(
                "AdamW.step(Collection<Tensor>) cannot replace immutable tensor values; "
                        + "use step(Map<String, Tensor>, Map<String, Tensor>)");
    }

    @Override
    public void step(Map<String, Tensor> params, Map<String, Tensor> grads) {
        Objects.requireNonNull(params, "params");
        Objects.requireNonNull(grads, "grads");

        // Advance bias correction only when at least one trainable parameter has a gradient.
        boolean hasGradient = params.entrySet().stream()
                .anyMatch(entry -> entry.getValue() != null
                        && entry.getValue().requiresGrad()
                        && grads.get(entry.getKey()) != null);
        if (!hasGradient) {
            return;
        }
        state.step++;

        for (Map.Entry<String, Tensor> entry : params.entrySet()) {
            String name = entry.getKey();
            Tensor parameter = Objects.requireNonNull(entry.getValue(), "parameter " + name);
            Tensor gradient = grads.get(name);
            if (gradient == null || !parameter.requiresGrad()) {
                continue;
            }

            state.m.putIfAbsent(name, gradient.zerosLike());
            state.v.putIfAbsent(name, gradient.zerosLike());

            Tensor m = state.m.get(name).mul(beta1).add(gradient.mul(1.0f - beta1));
            Tensor v = state.v.get(name).mul(beta2).add(gradient.mul(gradient).mul(1.0f - beta2));
            state.m.put(name, m);
            state.v.put(name, v);

            float bias1 = 1.0f - (float) Math.pow(beta1, state.step);
            float bias2 = 1.0f - (float) Math.pow(beta2, state.step);
            Tensor mHat = m.div(bias1);
            Tensor vHat = v.div(bias2);
            Tensor update = mHat.div(vHat.sqrt().add(eps));

            Tensor decayed = parameter.sub(parameter.mul(lr * weightDecay));
            params.put(name, decayed.sub(update.mul(lr)));
        }
    }
}
