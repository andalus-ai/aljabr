package tech.kayys.aljabr.diffusion.scheduler.ddim;

import java.util.Arrays;
import java.util.Objects;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionSchedule;

/** DDIM schedule retaining the training cumulative-alpha coefficients. */
public final class DDIMInferenceSchedule {

    private final DiffusionSchedule schedule;
    private final float[] alphasCumprod;

    DDIMInferenceSchedule(DiffusionSchedule schedule, float[] alphasCumprod) {
        this.schedule = Objects.requireNonNull(schedule, "schedule");
        this.alphasCumprod = Arrays.copyOf(alphasCumprod, alphasCumprod.length);
    }

    public DiffusionSchedule schedule() {
        return schedule;
    }

    public int trainingTimesteps() {
        return alphasCumprod.length;
    }

    public int inferenceSteps() {
        return schedule.points().size();
    }

    public int timestepAt(int stepIndex) {
        return Math.toIntExact(Math.round(schedule.points().get(stepIndex).modelTime()));
    }

    public int previousTimestepAt(int stepIndex) {
        return stepIndex + 1 < inferenceSteps() ? timestepAt(stepIndex + 1) : -1;
    }

    public float alphaCumprodAtTrainingTimestep(int timestep) {
        return alphasCumprod[timestep];
    }

    float alphaCumprodAtStep(int stepIndex) {
        return alphasCumprod[timestepAt(stepIndex)];
    }

    float previousAlphaCumprodAtStep(int stepIndex) {
        int previousTimestep = previousTimestepAt(stepIndex);
        return previousTimestep < 0 ? 1.0f : alphasCumprod[previousTimestep];
    }

    public int[] timesteps() {
        return schedule.points().stream()
                .mapToInt(point -> Math.toIntExact(Math.round(point.modelTime())))
                .toArray();
    }
}
