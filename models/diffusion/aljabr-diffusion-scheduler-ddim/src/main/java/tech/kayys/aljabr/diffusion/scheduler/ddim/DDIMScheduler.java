package tech.kayys.aljabr.diffusion.scheduler.ddim;

import java.util.EnumSet;
import java.util.Objects;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionSchedule;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionSchedulePoint;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionScheduler;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;
import tech.kayys.aljabr.diffusion.scheduler.api.ScheduleDomain;
import tech.kayys.aljabr.diffusion.scheduler.api.SchedulerDescriptor;

/**
 * DDIM scheduler with a linear beta training schedule.
 *
 * <p>The implementation is scalar-only and independent of tensor storage and
 * compute backends. Use {@link #createInferenceSchedule(int)} when applying
 * DDIM updates, because that type also retains cumulative-alpha coefficients.</p>
 */
public final class DDIMScheduler implements DiffusionScheduler {

    private static final SchedulerDescriptor DESCRIPTOR = new SchedulerDescriptor(
            "ddim",
            ScheduleDomain.DISCRETE_TIMESTEPS,
            EnumSet.of(PredictionType.EPSILON),
            true);

    private final int trainingTimesteps;
    private final float betaStart;
    private final float betaEnd;
    private final float[] alphasCumprod;

    public DDIMScheduler(int trainingTimesteps, float betaStart, float betaEnd) {
        if (trainingTimesteps < 2) {
            throw new IllegalArgumentException("trainingTimesteps must be at least 2");
        }
        if (!Float.isFinite(betaStart) || !Float.isFinite(betaEnd)
                || betaStart <= 0.0f || betaEnd <= 0.0f
                || betaStart >= 1.0f || betaEnd >= 1.0f
                || betaStart > betaEnd) {
            throw new IllegalArgumentException(
                    "beta values must satisfy 0 < betaStart <= betaEnd < 1");
        }
        this.trainingTimesteps = trainingTimesteps;
        this.betaStart = betaStart;
        this.betaEnd = betaEnd;
        this.alphasCumprod = buildAlphasCumprod();
    }

    public DDIMScheduler(int trainingTimesteps) {
        this(trainingTimesteps, 0.0001f, 0.02f);
    }

    @Override
    public SchedulerDescriptor descriptor() {
        return DESCRIPTOR;
    }

    @Override
    public DiffusionSchedule createSchedule(int inferenceSteps, PredictionType predictionType) {
        Objects.requireNonNull(predictionType, "predictionType");
        if (!DESCRIPTOR.supports(predictionType)) {
            throw new IllegalArgumentException(
                    "DDIM supports EPSILON prediction, not " + predictionType);
        }
        return createInferenceSchedule(inferenceSteps).schedule();
    }

    public DDIMInferenceSchedule createInferenceSchedule(int inferenceSteps) {
        if (inferenceSteps < 1 || inferenceSteps > trainingTimesteps) {
            throw new IllegalArgumentException(
                    "inferenceSteps must be between 1 and " + trainingTimesteps);
        }

        var points = new java.util.ArrayList<DiffusionSchedulePoint>(inferenceSteps);
        if (inferenceSteps == 1) {
            points.add(new DiffusionSchedulePoint(0, trainingTimesteps - 1, Double.NaN));
        } else {
            for (int i = 0; i < inferenceSteps; i++) {
                int ascending = Math.round(
                        (float) i * (trainingTimesteps - 1) / (inferenceSteps - 1));
                int timestep = trainingTimesteps - 1 - ascending;
                points.add(new DiffusionSchedulePoint(i, timestep, Double.NaN));
            }
        }
        DiffusionSchedule schedule = new DiffusionSchedule(
                ScheduleDomain.DISCRETE_TIMESTEPS, PredictionType.EPSILON, points);
        return new DDIMInferenceSchedule(schedule, alphasCumprod);
    }

    public DDIMStepParameters stepParameters(
            DDIMInferenceSchedule schedule, int stepIndex, float eta) {
        Objects.requireNonNull(schedule, "schedule");
        if (schedule.trainingTimesteps() != trainingTimesteps) {
            throw new IllegalArgumentException(
                    "schedule must be created for this scheduler's training timestep count");
        }
        if (stepIndex < 0 || stepIndex >= schedule.inferenceSteps()) {
            throw new IllegalArgumentException("stepIndex is outside the inference schedule");
        }
        if (!Float.isFinite(eta) || eta < 0.0f) {
            throw new IllegalArgumentException("eta must be finite and non-negative");
        }

        float alpha = schedule.alphaCumprodAtStep(stepIndex);
        float alphaPrevious = schedule.previousAlphaCumprodAtStep(stepIndex);
        float variance = ((1.0f - alphaPrevious) / (1.0f - alpha))
                * (1.0f - alpha / alphaPrevious);
        float sigma = eta * (float) Math.sqrt(Math.max(variance, 0.0f));
        float directionVariance = 1.0f - alphaPrevious - sigma * sigma;
        if (directionVariance < -1.0e-6f) {
            throw new IllegalArgumentException(
                    "eta is too large for a numerically valid DDIM update");
        }
        float directionCoefficient =
                (float) Math.sqrt(Math.max(directionVariance, 0.0f));

        return new DDIMStepParameters(
                schedule.timestepAt(stepIndex),
                schedule.previousTimestepAt(stepIndex),
                alpha,
                alphaPrevious,
                sigma,
                directionCoefficient);
    }

    public int trainingTimesteps() {
        return trainingTimesteps;
    }

    public float betaStart() {
        return betaStart;
    }

    public float betaEnd() {
        return betaEnd;
    }

    private float[] buildAlphasCumprod() {
        float[] result = new float[trainingTimesteps];
        float cumulative = 1.0f;
        for (int timestep = 0; timestep < trainingTimesteps; timestep++) {
            float fraction = (float) timestep / (trainingTimesteps - 1);
            float beta = betaStart + fraction * (betaEnd - betaStart);
            cumulative *= 1.0f - beta;
            result[timestep] = cumulative;
        }
        return result;
    }
}
