package tech.kayys.aljabr.diffusion.execution.api;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;

/**
 * Converts a denoiser output into the derivative representation required by
 * a sigma-space solver.
 *
 * <p>Implementations are model-parameterization aware. In particular,
 * {@link PredictionType#VELOCITY} is not assigned a universal conversion:
 * velocity definitions differ across training objectives and model families.</p>
 */
public interface SigmaPredictionConverter {

    /** Prediction semantics accepted by this converter. */
    PredictionType predictionType();

    /**
     * Produces the derivative for the current sample and model output.
     *
     * @param sample borrowed current noisy sample
     * @param modelOutput borrowed denoiser output
     * @param context immutable current/next sigma coordinates
     * @return a newly owned derivative tensor; must not alias either input
     */
    Tensor toDerivative(Tensor sample, Tensor modelOutput, SigmaStepContext context);
}
