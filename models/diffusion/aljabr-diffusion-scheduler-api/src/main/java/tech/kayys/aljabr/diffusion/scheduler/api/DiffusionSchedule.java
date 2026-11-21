package tech.kayys.aljabr.diffusion.scheduler.api;

import java.util.List;
import java.util.Objects;

/** Immutable ordered inference schedule independent of a particular algorithm. */
public record DiffusionSchedule(
        ScheduleDomain domain,
        PredictionType predictionType,
        List<DiffusionSchedulePoint> points) {

    public DiffusionSchedule {
        Objects.requireNonNull(domain, "domain");
        Objects.requireNonNull(predictionType, "predictionType");
        points = List.copyOf(Objects.requireNonNull(points, "points"));
        if (points.isEmpty()) {
            throw new IllegalArgumentException("schedule must contain at least one point");
        }
        for (int i = 0; i < points.size(); i++) {
            DiffusionSchedulePoint point = Objects.requireNonNull(points.get(i), "schedule point");
            if (point.index() != i) {
                throw new IllegalArgumentException("schedule point indexes must be contiguous from zero");
            }
            if ((domain == ScheduleDomain.CONTINUOUS_SIGMA) != point.hasSigma()) {
                throw new IllegalArgumentException(
                        "sigma must be present exactly for CONTINUOUS_SIGMA schedules");
            }
        }
    }
}
