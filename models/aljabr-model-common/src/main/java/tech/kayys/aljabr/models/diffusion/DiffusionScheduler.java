package tech.kayys.aljabr.models.diffusion;

/**
 * Produces backend-independent diffusion schedules and scalar reverse-step
 * coefficients. Tensor operations remain the responsibility of the execution
 * backend or denoising loop.
 */
public interface DiffusionScheduler {

    /**
     * Creates an inference schedule with the requested number of denoising steps.
     *
     * @param inferenceSteps number of reverse steps, between one and the
     *                       scheduler's training timestep count
     */
    DiffusionSchedule createSchedule(int inferenceSteps);

    /**
     * Calculates the scalar coefficients for one reverse step.
     *
     * @param schedule schedule returned by this scheduler
     * @param stepIndex zero-based index into the schedule's descending timestep list
     * @param eta stochasticity parameter; zero produces deterministic DDIM
     */
    SchedulerStepParameters stepParameters(
            DiffusionSchedule schedule, int stepIndex, float eta);
}
