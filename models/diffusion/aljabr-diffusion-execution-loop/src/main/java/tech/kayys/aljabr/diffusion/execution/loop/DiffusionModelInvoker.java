package tech.kayys.aljabr.diffusion.execution.loop;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.api.PreparedDiffusionConditioningSet;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/**
 * Model-family integration point for one diffusion network evaluation.
 *
 * <p>The invoker owns the architecture-specific mapping from prepared
 * conditioning data to model inputs. It may use any Aljabr model and backend;
 * this contract does not prescribe a tensor layout or conditioning format.</p>
 *
 * <p>The sample and conditioning set are borrowed for the duration of the
 * call. Implementations must return a newly owned prediction tensor and must
 * not close either borrowed input.</p>
 */
@FunctionalInterface
public interface DiffusionModelInvoker {
    Tensor predict(
            Tensor sample,
            PreparedDiffusionConditioningSet conditioning,
            SigmaStepContext context);
}
