package tech.kayys.aljabr.diffusion.execution.api;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Executes one scheduler-defined sigma-space tensor update.
 *
 * <p>Implementations own solver mathematics, not tensor storage or backend
 * selection. Inputs are borrowed and must remain open for the duration of the
 * call. The executor owns the derivative returned by the converter and must
 * close it on both success and failure. The returned sample is newly owned by
 * the caller.</p>
 */
public interface SigmaStepExecutor {

    /**
     * Applies one reverse step.
     *
     * @param sample borrowed current noisy sample
     * @param modelOutput borrowed model prediction for the current sigma
     * @param context current step coordinates and prediction semantics
     * @param converter model-family supplied prediction conversion
     * @return newly owned next sample tensor
     */
    Tensor execute(
            Tensor sample,
            Tensor modelOutput,
            SigmaStepContext context,
            SigmaPredictionConverter converter);
}
