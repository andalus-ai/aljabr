package tech.kayys.aljabr.data;

import tech.kayys.aljabr.core.tensor.Tensor;
/**
 * 
 * Core class for tech module.
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class Batch {
    public final Tensor tokens;
    public final Tensor targets;

    public Batch(Tensor tokens, Tensor targets) {
        this.tokens = tokens;
        this.targets = targets;
    }
}