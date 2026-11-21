package tech.kayys.aljabr.diffusion.scheduler.euler;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Objects;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionSchedule;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionSchedulePoint;
import tech.kayys.aljabr.diffusion.scheduler.api.DiffusionScheduler;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;
import tech.kayys.aljabr.diffusion.scheduler.api.ScheduleDomain;
import tech.kayys.aljabr.diffusion.scheduler.api.SchedulerDescriptor;

/**
 * Deterministic first-order Euler scheduler using a Karras rho sigma schedule.
 *
 * <p>This class constructs coordinates and scalar step deltas only. Conversion
 * from model prediction to the derivative, and tensor updates, belong to the
 * diffusion execution layer.</p>
 */
public final class EulerScheduler implements DiffusionScheduler {

    private static final SchedulerDescriptor DESCRIPTOR = new SchedulerDescriptor(
            "euler",
            ScheduleDomain.CONTINUOUS_SIGMA,
            EnumSet.of(PredictionType.EPSILON, PredictionType.VELOCITY, PredictionType.SAMPLE),
            false);

    private final double sigmaMin;
    private final double sigmaMax;
    private final double rho;

    public EulerScheduler(double sigmaMin, double sigmaMax, double rho) {
        if (!Double.isFinite(sigmaMin) || !Double.isFinite(sigmaMax)
                || sigmaMin <= 0.0d || sigmaMax <= sigmaMin) {
            throw new IllegalArgumentException("sigmas must satisfy 0 < sigmaMin < sigmaMax");
        }
        if (!Double.isFinite(rho) || rho <= 0.0d) {
            throw new IllegalArgumentException("rho must be finite and positive");
        }
        this.sigmaMin = sigmaMin;
        this.sigmaMax = sigmaMax;
        this.rho = rho;
    }

    public EulerScheduler(double sigmaMin, double sigmaMax) {
        this(sigmaMin, sigmaMax, 7.0d);
    }

    @Override
    public SchedulerDescriptor descriptor() {
        return DESCRIPTOR;
    }

    @Override
    public DiffusionSchedule createSchedule(int inferenceSteps, PredictionType predictionType) {
        return createInferenceSchedule(inferenceSteps, predictionType).schedule();
    }

    public EulerInferenceSchedule createInferenceSchedule(int inferenceSteps) {
        return createInferenceSchedule(inferenceSteps, PredictionType.EPSILON);
    }

    public EulerInferenceSchedule createInferenceSchedule(
            int inferenceSteps, PredictionType predictionType) {
        Objects.requireNonNull(predictionType, "predictionType");
        if (!DESCRIPTOR.supports(predictionType)) {
            throw new IllegalArgumentException("Euler does not support prediction type " + predictionType);
        }
        if (inferenceSteps < 1) {
            throw new IllegalArgumentException("inferenceSteps must be at least 1");
        }

        double minRoot = Math.pow(sigmaMin, 1.0d / rho);
        double maxRoot = Math.pow(sigmaMax, 1.0d / rho);
        var points = new ArrayList<DiffusionSchedulePoint>(inferenceSteps);
        for (int i = 0; i < inferenceSteps; i++) {
            double ramp = inferenceSteps == 1 ? 0.0d : (double) i / (inferenceSteps - 1);
            double sigma = Math.pow(maxRoot + ramp * (minRoot - maxRoot), rho);
            points.add(new DiffusionSchedulePoint(i, sigma, sigma));
        }

        DiffusionSchedule schedule = new DiffusionSchedule(
                ScheduleDomain.CONTINUOUS_SIGMA, predictionType, points);
        return new EulerInferenceSchedule(schedule, 0.0d);
    }

    public EulerStepParameters stepParameters(EulerInferenceSchedule schedule, int stepIndex) {
        Objects.requireNonNull(schedule, "schedule");
        if (stepIndex < 0 || stepIndex >= schedule.inferenceSteps()) {
            throw new IllegalArgumentException("stepIndex is outside the inference schedule");
        }
        double sigma = schedule.sigmaAt(stepIndex);
        double nextSigma = schedule.nextSigmaAt(stepIndex);
        return new EulerStepParameters(sigma, nextSigma, nextSigma - sigma);
    }

    public double sigmaMin() {
        return sigmaMin;
    }

    public double sigmaMax() {
        return sigmaMax;
    }

    public double rho() {
        return rho;
    }
}
