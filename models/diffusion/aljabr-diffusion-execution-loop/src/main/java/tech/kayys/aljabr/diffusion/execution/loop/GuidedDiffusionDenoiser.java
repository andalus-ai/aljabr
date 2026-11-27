package tech.kayys.aljabr.diffusion.execution.loop;

import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/**
 * Adapts classifier-free guidance to the denoiser callback consumed by the
 * shared diffusion loop and scheduler-specific runners.
 *
 * <p>This is the composition point between the iterative solver and CFG. A
 * model-family provider captures its conditional/unconditional conditioning
 * inputs and performs model evaluation; this adapter applies the configured
 * guidance and returns the resulting newly owned model prediction.</p>
 *
 * <p>The adapter is immutable and can be reused across all steps of one
 * generation request. It does not own the provider, its conditioning data, or
 * the guidance engine.</p>
 */
public final class GuidedDiffusionDenoiser implements DiffusionDenoiser {

    private final ClassifierFreeGuidance guidance;
    private final GuidancePredictionProvider provider;
    private final double guidanceScale;
    private final double guidanceRescale;

    public GuidedDiffusionDenoiser(
            ClassifierFreeGuidance guidance,
            GuidancePredictionProvider provider,
            double guidanceScale,
            double guidanceRescale) {
        this.guidance = Objects.requireNonNull(guidance, "guidance");
        this.provider = Objects.requireNonNull(provider, "provider");
        if (!Double.isFinite(guidanceScale) || guidanceScale < 0.0d) {
            throw new IllegalArgumentException(
                    "guidanceScale must be finite and non-negative");
        }
        if (!Double.isFinite(guidanceRescale)
                || guidanceRescale < 0.0d
                || guidanceRescale > 1.0d) {
            throw new IllegalArgumentException(
                    "guidanceRescale must be in the range [0, 1]");
        }
        this.guidanceScale = guidanceScale;
        this.guidanceRescale = guidanceRescale;
    }

    @Override
    public Tensor predict(Tensor sample, SigmaStepContext context) {
        return guidance.predict(
                sample, context, provider, guidanceScale, guidanceRescale);
    }

    public double guidanceScale() {
        return guidanceScale;
    }

    public double guidanceRescale() {
        return guidanceRescale;
    }
}
