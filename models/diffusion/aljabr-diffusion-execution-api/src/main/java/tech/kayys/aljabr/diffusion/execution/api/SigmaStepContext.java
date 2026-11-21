package tech.kayys.aljabr.diffusion.execution.api;

import java.util.Objects;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;

/**
 * Immutable coordinates and prediction semantics for one sigma-space solver step.
 *
 * <p>The current sigma must be positive because prediction-to-derivative
 * conversion may divide by it. The next sigma may be zero for the terminal
 * denoising transition, but cannot exceed the current sigma.</p>
 */
public record SigmaStepContext(
        int stepIndex,
        double sigma,
        double nextSigma,
        PredictionType predictionType) {

    public SigmaStepContext {
        if (stepIndex < 0) {
            throw new IllegalArgumentException("stepIndex must be non-negative");
        }
        if (!Double.isFinite(sigma) || sigma <= 0.0d) {
            throw new IllegalArgumentException("sigma must be finite and positive");
        }
        if (!Double.isFinite(nextSigma) || nextSigma < 0.0d || nextSigma > sigma) {
            throw new IllegalArgumentException(
                    "nextSigma must be finite and in the range [0, sigma]");
        }
        Objects.requireNonNull(predictionType, "predictionType");
    }

    /** Signed sigma interval; reverse denoising steps have a non-positive delta. */
    public double deltaSigma() {
        return nextSigma - sigma;
    }

    public boolean terminalStep() {
        return nextSigma == 0.0d;
    }
}
