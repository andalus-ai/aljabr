package tech.kayys.aljabr.diffusion.execution.loop;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.function.IntConsumer;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaPredictionConverter;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepExecutor;

/**
 * Executes an ordered sigma-space denoising sequence with explicit tensor
 * ownership, cancellation, and progress reporting.
 *
 * <p>This class deliberately does not construct schedules or choose model
 * architectures. The caller supplies the scheduler-derived step contexts and
 * family-specific prediction converter.</p>
 */
public final class DiffusionDenoisingLoop {

    private final SigmaStepExecutor stepExecutor;

    public DiffusionDenoisingLoop(SigmaStepExecutor stepExecutor) {
        this.stepExecutor = Objects.requireNonNull(stepExecutor, "stepExecutor");
    }

    /**
     * Runs all steps and returns a newly owned final sample.
     *
     * @param initialSample borrowed initial sample; never closed by this loop
     * @param steps ordered solver contexts, indexed from zero
     * @param denoiser model evaluation callback; each result must be newly owned
     * @param converter family-specific prediction conversion
     * @param cancelled cancellation predicate; may be {@code null} for no cancellation
     * @param progress completed-step callback; may be {@code null}
     * @return newly owned final sample; caller must close it
     */
    public Tensor run(
            Tensor initialSample,
            List<SigmaStepContext> steps,
            DiffusionDenoiser denoiser,
            SigmaPredictionConverter converter,
            BooleanSupplier cancelled,
            IntConsumer progress) {
        Objects.requireNonNull(initialSample, "initialSample");
        List<SigmaStepContext> orderedSteps = List.copyOf(
                Objects.requireNonNull(steps, "steps"));
        Objects.requireNonNull(denoiser, "denoiser");
        Objects.requireNonNull(converter, "converter");
        if (orderedSteps.isEmpty()) {
            throw new IllegalArgumentException("steps must not be empty");
        }
        for (int i = 0; i < orderedSteps.size(); i++) {
            SigmaStepContext context = Objects.requireNonNull(orderedSteps.get(i), "step");
            if (context.stepIndex() != i) {
                throw new IllegalArgumentException("step indexes must be contiguous from zero");
            }
            if (context.predictionType() != converter.predictionType()) {
                throw new IllegalArgumentException(
                        "Step prediction type " + context.predictionType()
                                + " does not match converter prediction type "
                                + converter.predictionType());
            }
        }

        Tensor current = initialSample;
        boolean ownsCurrent = false;
        try {
            for (SigmaStepContext context : orderedSteps) {
                if (cancelled != null && cancelled.getAsBoolean()) {
                    throw new CancellationException(
                            "Diffusion denoising cancelled before step " + context.stepIndex());
                }

                Tensor modelOutput = Objects.requireNonNull(
                        denoiser.predict(current, context),
                        "denoiser returned null model output");
                if (modelOutput == current) {
                    throw new IllegalArgumentException(
                            "Denoiser output must not alias the borrowed sample");
                }

                Tensor next;
                try (modelOutput) {
                    next = Objects.requireNonNull(
                            stepExecutor.execute(current, modelOutput, context, converter),
                            "step executor returned null sample");
                }
                if (next == current || next == modelOutput) {
                    // modelOutput is already closed by the resource scope; do not
                    // close current because it may be the caller's borrowed tensor.
                    throw new IllegalStateException(
                            "Step executor must return a newly owned sample tensor");
                }

                if (ownsCurrent) {
                    current.close();
                }
                current = next;
                ownsCurrent = true;

                if (progress != null) {
                    progress.accept(context.stepIndex() + 1);
                }
            }
            Tensor result = current;
            ownsCurrent = false;
            return result;
        } finally {
            if (ownsCurrent) {
                current.close();
            }
        }
    }
}
