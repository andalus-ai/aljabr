package tech.kayys.aljabr.diffusion.scheduler.api;

/**
 * One position in an inference schedule. The coordinate is interpreted using
 * the owning schedule's domain; sigma is optional and represented by NaN when
 * the domain does not use sigma coordinates.
 */
public record DiffusionSchedulePoint(int index, double modelTime, double sigma) {

    public DiffusionSchedulePoint {
        if (index < 0) {
            throw new IllegalArgumentException("index must be non-negative");
        }
        if (!Double.isFinite(modelTime)) {
            throw new IllegalArgumentException("modelTime must be finite");
        }
        if (!Double.isNaN(sigma) && (!Double.isFinite(sigma) || sigma < 0.0d)) {
            throw new IllegalArgumentException("sigma must be non-negative and finite, or NaN when absent");
        }
    }

    public boolean hasSigma() {
        return !Double.isNaN(sigma);
    }
}
