package tech.kayys.aljabr.models.diffusion;

import java.util.Arrays;
import java.util.Objects;

/**
 * Compatibility adapter for the modular DDIM scheduler.
 *
 * <p>This legacy package remains available to existing pipeline consumers.
 * All schedule construction and DDIM coefficient mathematics are delegated
 * to the independently installable diffusion scheduler module.</p>
 */
@Deprecated(since = "0.2.0", forRemoval = false)
public final class DDIMScheduler implements DiffusionScheduler {

    private final tech.kayys.aljabr.diffusion.scheduler.ddim.DDIMScheduler delegate;
    private final int trainingTimesteps;

    public DDIMScheduler(int trainingTimesteps, float betaStart, float betaEnd) {
        this.delegate = new tech.kayys.aljabr.diffusion.scheduler.ddim.DDIMScheduler(
                trainingTimesteps, betaStart, betaEnd);
        this.trainingTimesteps = trainingTimesteps;
    }

    public DDIMScheduler(int trainingTimesteps) {
        this(trainingTimesteps, 0.0001f, 0.02f);
    }

    @Override
    public DiffusionSchedule createSchedule(int inferenceSteps) {
        var modularSchedule = delegate.createInferenceSchedule(inferenceSteps);
        float[] alphasCumprod = new float[trainingTimesteps];
        for (int timestep = 0; timestep < trainingTimesteps; timestep++) {
            alphasCumprod[timestep] =
                    modularSchedule.alphaCumprodAtTrainingTimestep(timestep);
        }
        return new DiffusionSchedule(alphasCumprod, modularSchedule.timesteps());
    }

    @Override
    public SchedulerStepParameters stepParameters(
            DiffusionSchedule schedule, int stepIndex, float eta) {
        Objects.requireNonNull(schedule, "schedule");
        if (schedule.trainingTimesteps() != trainingTimesteps) {
            throw new IllegalArgumentException(
                    "schedule must be created for this scheduler's training timestep count");
        }

        var modularSchedule = delegate.createInferenceSchedule(schedule.inferenceSteps());
        if (!Arrays.equals(schedule.timesteps(), modularSchedule.timesteps())) {
            throw new IllegalArgumentException(
                    "schedule timesteps do not match this scheduler's inference schedule");
        }

        var parameters = delegate.stepParameters(modularSchedule, stepIndex, eta);
        return new SchedulerStepParameters(
                parameters.timestep(),
                parameters.previousTimestep(),
                parameters.alphaCumprod(),
                parameters.previousAlphaCumprod(),
                parameters.sigma(),
                parameters.directionCoefficient());
    }

    public int trainingTimesteps() {
        return delegate.trainingTimesteps();
    }

    public float betaStart() {
        return delegate.betaStart();
    }

    public float betaEnd() {
        return delegate.betaEnd();
    }
}
