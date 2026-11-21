package tech.kayys.aljabr.diffusion.scheduler.api;

/**
 * Scheduler contract for constructing inference coordinates and declaring
 * supported prediction semantics. Tensor update execution is deliberately
 * added through a separate execution contract once tensor ownership and
 * multistep scheduler state are finalized.
 */
public interface DiffusionScheduler {

    SchedulerDescriptor descriptor();

    DiffusionSchedule createSchedule(int inferenceSteps, PredictionType predictionType);
}
