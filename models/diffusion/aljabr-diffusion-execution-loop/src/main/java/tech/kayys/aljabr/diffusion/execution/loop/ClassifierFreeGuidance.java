package tech.kayys.aljabr.diffusion.execution.loop;

import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;

/**
 * Reusable classifier-free guidance orchestration for diffusion models.
 *
 * <p>This class owns neither conditioning objects nor model/backend selection.
 * A model-family adapter supplies predictions; a backend-aware tensor
 * operations adapter supplies the algebra. Batched prediction is preferred
 * when explicitly advertised, with a sequential fallback.</p>
 */
public final class ClassifierFreeGuidance {

    private final GuidanceTensorOperations tensorOperations;

    public ClassifierFreeGuidance(GuidanceTensorOperations tensorOperations) {
        this.tensorOperations = Objects.requireNonNull(tensorOperations, "tensorOperations");
    }

    /**
     * Produces one newly owned guided prediction.
     *
     * @param sample borrowed latent/sample tensor
     * @param context current scheduler coordinates
     * @param provider model-family prediction adapter
     * @param guidanceScale CFG scale; must be finite and non-negative
     * @param guidanceRescale optional standard-deviation rescale in [0, 1]
     * @return newly owned guided prediction; caller must close it
     */
    public Tensor predict(
            Tensor sample,
            SigmaStepContext context,
            GuidancePredictionProvider provider,
            double guidanceScale,
            double guidanceRescale) {
        Objects.requireNonNull(sample, "sample");
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(provider, "provider");
        validateGuidance(guidanceScale, guidanceRescale);

        GuidancePredictions predictions = provider.supportsBatchedPrediction()
                ? Objects.requireNonNull(
                        provider.predictBatched(sample, context),
                        "batched provider returned null predictions")
                : predictSequential(sample, context, provider);

        try (predictions) {
            if (predictions.unconditional() == sample
                    || predictions.conditional() == sample) {
                throw new IllegalStateException(
                        "Guidance predictions must not alias the borrowed sample");
            }
            Tensor guided = Objects.requireNonNull(
                    tensorOperations.combine(
                            predictions.unconditional(),
                            predictions.conditional(),
                            guidanceScale),
                    "guidance combine returned null tensor");
            if (guided == sample
                    || guided == predictions.unconditional()
                    || guided == predictions.conditional()) {
                throw new IllegalStateException(
                        "Guidance combine must return a newly owned tensor");
            }

            if (guidanceRescale == 0.0d) {
                return guided;
            }

            try {
                Tensor rescaled = Objects.requireNonNull(
                        tensorOperations.rescale(
                                guided, predictions.conditional(), guidanceRescale),
                        "guidance rescale returned null tensor");
                if (rescaled == guided
                        || rescaled == sample
                        || rescaled == predictions.unconditional()
                        || rescaled == predictions.conditional()) {
                    throw new IllegalStateException(
                            "Guidance rescale must return a newly owned tensor");
                }
                return rescaled;
            } finally {
                guided.close();
            }
        }
    }

    private static GuidancePredictions predictSequential(
            Tensor sample,
            SigmaStepContext context,
            GuidancePredictionProvider provider) {
        Tensor unconditional = Objects.requireNonNull(
                provider.predictUnconditional(sample, context),
                "unconditional prediction returned null");
        if (unconditional == sample) {
            throw new IllegalStateException(
                    "Unconditional prediction must not alias the borrowed sample");
        }

        try {
            Tensor conditional = Objects.requireNonNull(
                    provider.predictConditional(sample, context),
                    "conditional prediction returned null");
            if (conditional == sample || conditional == unconditional) {
                throw new IllegalStateException(
                        "Conditional prediction must be distinct and newly owned");
            }
            return new GuidancePredictions(unconditional, conditional);
        } catch (RuntimeException | Error failure) {
            unconditional.close();
            throw failure;
        }
    }

    private static void validateGuidance(double scale, double rescale) {
        if (!Double.isFinite(scale) || scale < 0.0d) {
            throw new IllegalArgumentException("guidanceScale must be finite and non-negative");
        }
        if (!Double.isFinite(rescale) || rescale < 0.0d || rescale > 1.0d) {
            throw new IllegalArgumentException("guidanceRescale must be in the range [0, 1]");
        }
    }
}
