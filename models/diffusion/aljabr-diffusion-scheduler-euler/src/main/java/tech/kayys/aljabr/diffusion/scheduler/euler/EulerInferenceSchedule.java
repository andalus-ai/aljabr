package tech.kayys.aljabr.diffusion.scheduler.euler;

import java.util.Objects;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionSchedule;

/** Euler schedule retaining the terminal sigma used after the final model evaluation. */
public final class EulerInferenceSchedule {

    private final DiffusionSchedule schedule;
    private final double terminalSigma;

    EulerInferenceSchedule(DiffusionSchedule schedule, double terminalSigma) {
        this.schedule = Objects.requireNonNull(schedule, "schedule");
        if (!Double.isFinite(terminalSigma) || terminalSigma < 0.0d) {
            throw new IllegalArgumentException("terminalSigma must be finite and non-negative");
        }
        this.terminalSigma = terminalSigma;
    }

    public DiffusionSchedule schedule() {
        return schedule;
    }

    public int inferenceSteps() {
        return schedule.points().size();
    }

    public double sigmaAt(int stepIndex) {
        return schedule.points().get(stepIndex).sigma();
    }

    public double nextSigmaAt(int stepIndex) {
        return stepIndex + 1 < inferenceSteps()
                ? sigmaAt(stepIndex + 1)
                : terminalSigma;
    }

    public double terminalSigma() {
        return terminalSigma;
    }
}
