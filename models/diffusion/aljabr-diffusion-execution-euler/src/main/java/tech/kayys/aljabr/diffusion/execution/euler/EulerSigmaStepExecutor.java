package tech.kayys.aljabr.diffusion.execution.euler;

import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaPredictionConverter;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepExecutor;

/**
 * Stateless explicit Euler integrator for a reverse sigma-space transition.
 *
 * <p>Given a model-family supplied derivative {@code d}, this executor applies
 * {@code sample + d * (nextSigma - sigma)}. It does not decide how a denoiser
 * prediction maps to {@code d}; that responsibility belongs to the converter.</p>
 */
public final class EulerSigmaStepExecutor implements SigmaStepExecutor {

    @Override
    public Tensor execute(
            Tensor sample,
            Tensor modelOutput,
            SigmaStepContext context,
            SigmaPredictionConverter converter) {
        Objects.requireNonNull(sample, "sample");
        Objects.requireNonNull(modelOutput, "modelOutput");
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(converter, "converter");

        if (converter.predictionType() != context.predictionType()) {
            throw new IllegalArgumentException(
                    "Converter prediction type " + converter.predictionType()
                            + " does not match step prediction type " + context.predictionType());
        }

        if (!sample.shape().equals(modelOutput.shape())) {
            throw new IllegalArgumentException(
                    "Model output shape " + modelOutput.shape()
                            + " does not match sample shape " + sample.shape());
        }
        if (sample.backend() != modelOutput.backend()) {
            throw new IllegalArgumentException(
                    "Sample and model output must use the same compute backend");
        }

        float deltaSigma = (float) context.deltaSigma();
        if (!Float.isFinite(deltaSigma)) {
            throw new IllegalArgumentException(
                    "Sigma delta cannot be represented by the tensor scalar API: "
                            + context.deltaSigma());
        }

        Tensor derivative = Objects.requireNonNull(
                converter.toDerivative(sample, modelOutput, context),
                "converter returned a null derivative");
        try (derivative) {
            if (!sample.shape().equals(derivative.shape())) {
                throw new IllegalArgumentException(
                        "Derivative shape " + derivative.shape()
                                + " does not match sample shape " + sample.shape());
            }
            if (sample.backend() != derivative.backend()) {
                throw new IllegalArgumentException(
                        "Derivative must use the same compute backend as the sample");
            }
            try (Tensor scaledDerivative = derivative.mul(deltaSigma)) {
                return sample.add(scaledDerivative);
            }
        }
    }
}
