package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.Map;

/**
 * GRADIENT SCALING (CRITICAL)
 * Problem:
 * FP16 gradients → underflow → zero ❌
 * Solution:
 * scale loss → compute grads → unscale → update
 *
 * registry.register("gradscaler", new GradScaler());
 * @author bhangun
 */
public final class GradScaler {
    private float scale = 1024.0f;
    private final float growth = 2.0f;
    private final float backoff = 0.5f;

    public float scale() {
        return scale;
    }

    public void update(boolean overflow) {
        if (overflow) {
            scale *= backoff;
        } else {
            scale *= growth;
        }
    }

    public Tensor scale(Tensor t) {
        return t.mul(scale);
    }

    public void unscale(Map<String, Tensor> grads) {
        for (String k : grads.keySet()) {
            grads.put(k, grads.get(k).div(scale));
        }
    }
}