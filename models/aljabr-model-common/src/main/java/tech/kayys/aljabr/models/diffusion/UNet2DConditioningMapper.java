package tech.kayys.aljabr.models.diffusion;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.api.PreparedDiffusionConditioningSet;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/**
 * Family/checkpoint-specific mapping from generic diffusion inputs to the
 * timestep and encoder tensors consumed by UNet2DConditional.
 *
 * <p>Implementations own tokenizer/encoder output selection, sigma-to-time
 * conversion, batching, and any checkpoint-specific conditioning policy.
 * Returned tensors are newly owned by the caller.</p>
 */
@FunctionalInterface
public interface UNet2DConditioningMapper {
    UNet2DModelInputs map(
            Tensor latentSample,
            PreparedDiffusionConditioningSet conditioning,
            SigmaStepContext step);
}
