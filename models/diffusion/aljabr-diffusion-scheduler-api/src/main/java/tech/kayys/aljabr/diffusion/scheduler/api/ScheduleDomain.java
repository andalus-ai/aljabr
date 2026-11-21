package tech.kayys.aljabr.diffusion.scheduler.api;

/** Mathematical coordinate system used by a scheduler's inference schedule. */
public enum ScheduleDomain {
    /** Integer-indexed training/noise timesteps, typically based on alpha/beta values. */
    DISCRETE_TIMESTEPS,

    /** Continuous noise levels represented by sigma values. */
    CONTINUOUS_SIGMA,

    /** Continuous time parameterization used by flow-matching models. */
    CONTINUOUS_FLOW_TIME
}
