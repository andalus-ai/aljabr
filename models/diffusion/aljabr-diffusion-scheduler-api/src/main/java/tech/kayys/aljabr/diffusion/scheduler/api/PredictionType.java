package tech.kayys.aljabr.diffusion.scheduler.api;

/** Interpretation of a denoiser's output; families must declare this explicitly. */
public enum PredictionType {
    EPSILON,
    VELOCITY,
    SAMPLE,
    FLOW
}
