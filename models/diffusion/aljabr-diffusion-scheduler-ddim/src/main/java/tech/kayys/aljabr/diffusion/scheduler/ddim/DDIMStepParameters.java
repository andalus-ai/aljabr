package tech.kayys.aljabr.diffusion.scheduler.ddim;

/**
 * Scalar coefficients for a DDIM reverse update. A tensor execution layer
 * applies these values to samples, model predictions, and optional noise.
 */
public record DDIMStepParameters(
        int timestep,
        int previousTimestep,
        float alphaCumprod,
        float previousAlphaCumprod,
        float sigma,
        float directionCoefficient) {
}
