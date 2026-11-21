package tech.kayys.aljabr.models.diffusion;

/**
 * Scalar coefficients for a DDIM reverse update:
 *
 * <pre>
 * predictedOriginal = (sample - sqrt(1 - alpha) * epsilon) / sqrt(alpha)
 * previousSample = sqrt(alphaPrevious) * predictedOriginal
 *                + directionCoefficient * epsilon + sigma * noise
 * </pre>
 *
 * A backend can apply these coefficients to tensors without moving scheduler
 * mathematics or tensor data onto the host.
 */
public record SchedulerStepParameters(
        int timestep,
        int previousTimestep,
        float alphaCumprod,
        float previousAlphaCumprod,
        float sigma,
        float directionCoefficient) {
}
