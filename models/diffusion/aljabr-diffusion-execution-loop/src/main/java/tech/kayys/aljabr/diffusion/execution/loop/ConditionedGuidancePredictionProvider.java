package tech.kayys.aljabr.diffusion.execution.loop;

import java.util.Objects;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.api.PreparedDiffusionConditioningSet;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/**
 * Uses one model-family invoker for CFG's unconditional and conditional
 * branches, each with its own request-prepared conditioning set.
 *
 * <p>This implementation uses sequential evaluation. Families/backends that
 * can efficiently concatenate branch inputs may implement
 * {@link GuidancePredictionProvider} directly and opt into batched prediction.</p>
 */
public final class ConditionedGuidancePredictionProvider
        implements GuidancePredictionProvider {
    private final DiffusionModelInvoker model;
    private final PreparedDiffusionConditioningSet unconditional;
    private final PreparedDiffusionConditioningSet conditional;

    public ConditionedGuidancePredictionProvider(
            DiffusionModelInvoker model,
            PreparedDiffusionConditioningSet unconditional,
            PreparedDiffusionConditioningSet conditional) {
        this.model = Objects.requireNonNull(model, "model");
        this.unconditional = requireOpen(unconditional, "unconditional");
        this.conditional = requireOpen(conditional, "conditional");
    }

    @Override
    public Tensor predictUnconditional(Tensor sample, SigmaStepContext context) {
        return invoke(sample, unconditional, context);
    }

    @Override
    public Tensor predictConditional(Tensor sample, SigmaStepContext context) {
        return invoke(sample, conditional, context);
    }

    private Tensor invoke(
            Tensor sample,
            PreparedDiffusionConditioningSet conditioning,
            SigmaStepContext context) {
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

    private static PreparedDiffusionConditioningSet requireOpen(
            PreparedDiffusionConditioningSet conditioning, String name) {
        Objects.requireNonNull(conditioning, name);
        if (conditioning.isClosed()) {
            throw new IllegalArgumentException(name + " conditioning must be open");
        }
        return conditioning;
    }
}
