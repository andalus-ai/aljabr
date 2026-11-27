package tech.kayys.aljabr.diffusion.execution.euler;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.function.IntConsumer;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaPredictionConverter;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepExecutor;
import tech.kayys.aljabr.diffusion.execution.loop.DiffusionDenoiser;
import tech.kayys.aljabr.diffusion.execution.loop.DiffusionDenoisingLoop;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;
import tech.kayys.aljabr.diffusion.scheduler.euler.EulerInferenceSchedule;
import tech.kayys.aljabr.diffusion.scheduler.euler.EulerScheduler;

/**
 * Connects the Karras Euler inference schedule to the reusable sigma-space
 * denoising loop.
 *
 * <p>This is an orchestration adapter only. It does not know about checkpoint
 * formats, model architectures, conditioning, device selection, or prediction
 * parameterizations. Those remain the responsibility of the caller and model
 * family.</p>
 */
public final class EulerDenoisingRunner {

    private final EulerScheduler scheduler;
    private final DiffusionDenoisingLoop denoisingLoop;

    public EulerDenoisingRunner(EulerScheduler scheduler, SigmaStepExecutor stepExecutor) {
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
        this.denoisingLoop = new DiffusionDenoisingLoop(
                Objects.requireNonNull(stepExecutor, "stepExecutor"));
    }

    /**
     * Runs an Euler schedule and returns a newly owned final sample.
     *
     * @param initialSample borrowed starting latent/sample
     * @param inferenceSteps number of model evaluations
     * @param predictionType model output semantics
     * @param denoiser model-family inference callback; returns a newly owned tensor
     * @param converter model-family prediction-to-derivative conversion
     * @param cancelled optional cooperative cancellation predicate
     * @param progress optional completed-step callback
     * @return newly owned final sample; caller must close it
     * @throws CancellationException when cancellation is requested between steps
     */
    public Tensor run(
            Tensor initialSample,
            int inferenceSteps,
            PredictionType predictionType,
            DiffusionDenoiser denoiser,
            SigmaPredictionConverter converter,
            BooleanSupplier cancelled,
            IntConsumer progress) {
        Objects.requireNonNull(predictionType, "predictionType");
        Objects.requireNonNull(converter, "converter");
        if (converter.predictionType() != predictionType) {
            throw new IllegalArgumentException(
                    "Converter prediction type " + converter.predictionType()
                            + " does not match requested prediction type " + predictionType);
        }

        EulerInferenceSchedule schedule =
                scheduler.createInferenceSchedule(inferenceSteps, predictionType);
        return denoisingLoop.run(
                initialSample,
                toStepContexts(schedule, predictionType),
                denoiser,
                converter,
                cancelled,
                progress);
    }

    /**
     * Converts the scheduler's ordered coordinates, including its terminal
     * sigma, into the execution contract consumed by sigma-space solvers.
     */
    public static List<SigmaStepContext> toStepContexts(
            EulerInferenceSchedule schedule, PredictionType predictionType) {
        Objects.requireNonNull(schedule, "schedule");
        Objects.requireNonNull(predictionType, "predictionType");
        if (schedule.schedule().predictionType() != predictionType) {
            throw new IllegalArgumentException(
                    "Requested prediction type does not match scheduler schedule");
        }

        List<SigmaStepContext> contexts =
                new ArrayList<>(schedule.inferenceSteps());
        for (int i = 0; i < schedule.inferenceSteps(); i++) {
            contexts.add(new SigmaStepContext(
                    i,
                    schedule.sigmaAt(i),
                    schedule.nextSigmaAt(i),
                    predictionType));
        }
        return List.copyOf(contexts);
    }
}
