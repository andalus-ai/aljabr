package tech.kayys.aljabr.diffusion.scheduler.euler;

/** Scalar coordinates for one deterministic Euler reverse update. */
public record EulerStepParameters(
        double sigma,
        double nextSigma,
        double deltaSigma) {
}
