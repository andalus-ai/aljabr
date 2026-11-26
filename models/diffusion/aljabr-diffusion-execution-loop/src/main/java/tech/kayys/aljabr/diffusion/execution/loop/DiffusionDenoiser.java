package tech.kayys.aljabr.diffusion.execution.loop;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/**
 * Evaluates a diffusion model at one solver coordinate.
 *
 * <p>The sample is borrowed. The returned model-output tensor must be newly
 * owned by the caller of this method.</p>
 */
@FunctionalInterface
public interface DiffusionDenoiser {

    Tensor predict(Tensor sample, SigmaStepContext context);
}
