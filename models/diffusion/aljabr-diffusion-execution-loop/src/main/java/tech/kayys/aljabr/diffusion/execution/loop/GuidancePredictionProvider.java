package tech.kayys.aljabr.diffusion.execution.loop;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/**
 * Model-family adapter that evaluates unconditional and conditional
 * predictions for one diffusion step.
 *
 * <p>Conditioning representation and model invocation remain outside the
 * execution engine. Implementations may batch both branches into one model
 * invocation when their model and backend support it.</p>
 */
public interface GuidancePredictionProvider {

    /** Returns true only when {@link #predictBatched} is implemented. */
    default boolean supportsBatchedPrediction() {
        return false;
    }

    /**
     * Evaluates both branches in one model invocation.
     *
     * @return two newly owned, distinct prediction tensors
     */
    default GuidancePredictions predictBatched(Tensor sample, SigmaStepContext context) {
        throw new UnsupportedOperationException("Batched guidance is not supported");
    }

    /** Evaluates the unconditional branch and returns a newly owned tensor. */
    Tensor predictUnconditional(Tensor sample, SigmaStepContext context);

    /** Evaluates the conditional branch and returns a newly owned tensor. */
    Tensor predictConditional(Tensor sample, SigmaStepContext context);
}
