package tech.kayys.aljabr.diffusion.execution.loop;

import java.util.Objects;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.api.PreparedDiffusionConditioningSet;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/** Binds request-prepared conditioning to a reusable model-family invoker. */
public final class ConditionedDiffusionDenoiser implements DiffusionDenoiser {
    private final DiffusionModelInvoker model;
    private final PreparedDiffusionConditioningSet conditioning;

    public ConditionedDiffusionDenoiser(
            DiffusionModelInvoker model,
            PreparedDiffusionConditioningSet conditioning) {
        this.model = Objects.requireNonNull(model, "model");
        this.conditioning = requireOpen(conditioning);
    }

    @Override
    public Tensor predict(Tensor sample, SigmaStepContext context) {
        Objects.requireNonNull(sample, "sample");
        Objects.requireNonNull(context, "context");
        if (conditioning.isClosed()) {
            throw new IllegalStateException("conditioning was closed before model invocation");
        }
        Tensor prediction = Objects.requireNonNull(
                model.predict(sample, conditioning, context),
                "model invoker returned null prediction");
        if (prediction == sample) {
            throw new IllegalStateException(
                    "model invoker must return a newly owned prediction tensor");
        }
        return prediction;
    }

    /** Returns the borrowed request-owned conditioning set. */
    public PreparedDiffusionConditioningSet conditioning() {
        return conditioning;
    }

    private static PreparedDiffusionConditioningSet requireOpen(
            PreparedDiffusionConditioningSet conditioning) {
        Objects.requireNonNull(conditioning, "conditioning");
        if (conditioning.isClosed()) {
            throw new IllegalArgumentException("conditioning must be open");
        }
        return conditioning;
    }
}
