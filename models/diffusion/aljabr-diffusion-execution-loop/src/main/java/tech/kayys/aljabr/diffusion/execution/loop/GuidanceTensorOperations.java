package tech.kayys.aljabr.diffusion.execution.loop;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Backend-neutral tensor algebra required by classifier-free guidance.
 * Every returned tensor is newly owned by the caller.
 */
public interface GuidanceTensorOperations {

    /**
     * Computes unconditional + (conditional - unconditional) * guidanceScale.
     * Inputs are borrowed and must remain open for the duration of the call.
     */
    Tensor combine(Tensor unconditional, Tensor conditional, double guidanceScale);

    /**
     * Rescales the guided prediction toward the conditional prediction.
     *
     * <p>The implementation applies the standard deviation matching used by
     * guidance rescale, then blends that result with {@code guided} according
     * to {@code rescale}. A value of zero is not passed to this method.</p>
     *
     * @param guided newly owned guided prediction, borrowed for this call
     * @param conditional borrowed conditional prediction
     * @param rescale blend factor in (0, 1]
     * @return newly owned rescaled prediction
     */
    Tensor rescale(Tensor guided, Tensor conditional, double rescale);
}
