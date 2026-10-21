package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.Collection;
import java.util.Map;

/**
 * Stochastic Gradient Descent optimizer.
 * @author bhangun
 */
public final class SGD implements Optimizer {
    private final float lr;

    public SGD(float lr) {
        this.lr = lr;
    }

    @Override
    public void step(Collection<Tensor> params) {
        for (Tensor p : params) {
            if (p.requiresGrad() && p.grad() != null) {
                // In-place update logic would go here if backend supports it.
                // For now, we simulate the update.
                Tensor grad = p.grad();
                // p = p - lr * grad
                // This is a placeholder for actual in-place logic which requires buffer
                // manipulation
                // In this version, we just ensure the code compiles and follows the logic.
            }
        }
    }

    @Override
    public void step(Map<String, Tensor> params, Map<String, Tensor> grads) {
        for (String k : params.keySet()) {
            Tensor p = params.get(k);
            Tensor g = grads.get(k);
            if (g == null)
                continue;
            // Note: Tensor.sub returns a new tensor, so we update the map
            params.put(k, p.sub(g.mul(lr)));
        }
    }
}