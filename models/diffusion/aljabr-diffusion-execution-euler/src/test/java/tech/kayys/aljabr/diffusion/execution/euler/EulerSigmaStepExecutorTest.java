package tech.kayys.aljabr.diffusion.execution.euler;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.diffusion.execution.api.SigmaPredictionConverter;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;

class EulerSigmaStepExecutorTest {

    private final EulerSigmaStepExecutor executor = new EulerSigmaStepExecutor();

    @Test
    void appliesDerivativeUsingSignedSigmaDeltaWithoutMutatingBorrowedInputs() {
        try (CpuBackend backend = new CpuBackend();
                Tensor sample = TensorFactory.of(new float[] {10f, 20f}, backend, 2);
                Tensor modelOutput = TensorFactory.of(new float[] {3f, 4f}, backend, 2)) {
            SigmaStepContext context =
                    new SigmaStepContext(0, 4.0d, 2.0d, PredictionType.FLOW);
            SigmaPredictionConverter converter = converter(
                    PredictionType.FLOW, (current, output, step) -> current.mul(0.5f));

            try (Tensor result = executor.execute(sample, modelOutput, context, converter)) {
                assertArrayEquals(new float[] {0f, 0f}, result.toFloatArray(), 0.0001f);
            }

            assertArrayEquals(new float[] {10f, 20f}, sample.toFloatArray(), 0f);
            assertArrayEquals(new float[] {3f, 4f}, modelOutput.toFloatArray(), 0f);
        }
    }

    @Test
    void supportsTerminalTransitionToZeroSigma() {
        try (CpuBackend backend = new CpuBackend();
                Tensor sample = TensorFactory.of(new float[] {8f}, backend, 1);
                Tensor modelOutput = TensorFactory.of(new float[] {0f}, backend, 1)) {
            SigmaStepContext context =
                    new SigmaStepContext(1, 2.0d, 0.0d, PredictionType.FLOW);
            SigmaPredictionConverter converter = converter(
                    PredictionType.FLOW, (current, output, step) -> TensorFactory.of(
                            new float[] {1.5f}, backend, 1));

            try (Tensor result = executor.execute(sample, modelOutput, context, converter)) {
                assertArrayEquals(new float[] {5f}, result.toFloatArray(), 0.0001f);
            }
        }
    }

    @Test
    void rejectsConverterWithDifferentPredictionSemanticsBeforeConversion() {
        try (CpuBackend backend = new CpuBackend();
                Tensor sample = TensorFactory.of(new float[] {1f}, backend, 1);
                Tensor modelOutput = TensorFactory.of(new float[] {1f}, backend, 1)) {
            AtomicBoolean conversionCalled = new AtomicBoolean();
            SigmaPredictionConverter converter = converter(PredictionType.EPSILON,
                    (current, output, step) -> {
                        conversionCalled.set(true);
                        return current.mul(1f);
                    });
            SigmaStepContext context =
                    new SigmaStepContext(0, 1.0d, 0.5d, PredictionType.FLOW);

            assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(sample, modelOutput, context, converter));
            assertTrue(!conversionCalled.get());
        }
    }

    @Test
    void rejectsModelOutputWithShapeDifferentFromSampleBeforeConversion() {
        try (CpuBackend backend = new CpuBackend();
                Tensor sample = TensorFactory.of(new float[] {1f, 2f}, backend, 2);
                Tensor modelOutput = TensorFactory.of(new float[] {0f}, backend, 1)) {
            AtomicBoolean conversionCalled = new AtomicBoolean();
            SigmaPredictionConverter converter = converter(
                    PredictionType.FLOW, (current, output, step) -> {
                        conversionCalled.set(true);
                        return current.mul(1f);
                    });
            SigmaStepContext context =
                    new SigmaStepContext(0, 1.0d, 0.5d, PredictionType.FLOW);

            assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(sample, modelOutput, context, converter));
            assertTrue(!conversionCalled.get());
            assertArrayEquals(new float[] {1f, 2f}, sample.toFloatArray(), 0f);
        }
    }

    @Test
    void rejectsSampleAndModelOutputFromDifferentBackendsBeforeConversion() {
        try (CpuBackend sampleBackend = new CpuBackend();
                CpuBackend outputBackend = new CpuBackend();
                Tensor sample = TensorFactory.of(new float[] {1f, 2f}, sampleBackend, 2);
                Tensor modelOutput = TensorFactory.of(new float[] {0f, 0f}, outputBackend, 2)) {
            AtomicBoolean conversionCalled = new AtomicBoolean();
            SigmaPredictionConverter converter = converter(
                    PredictionType.FLOW, (current, output, step) -> {
                        conversionCalled.set(true);
                        return current.mul(1f);
                    });
            SigmaStepContext context =
                    new SigmaStepContext(0, 1.0d, 0.5d, PredictionType.FLOW);

            assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(sample, modelOutput, context, converter));
            assertTrue(!conversionCalled.get());
        }
    }

    @Test
    void rejectsDerivativeOwnedByDifferentBackend() {
        try (CpuBackend sampleBackend = new CpuBackend();
                CpuBackend derivativeBackend = new CpuBackend();
                Tensor sample = TensorFactory.of(new float[] {1f, 2f}, sampleBackend, 2);
                Tensor modelOutput = TensorFactory.of(new float[] {0f, 0f}, sampleBackend, 2)) {
            SigmaPredictionConverter converter = converter(
                    PredictionType.FLOW,
                    (current, output, step) ->
                            TensorFactory.of(new float[] {1f, 1f}, derivativeBackend, 2));
            SigmaStepContext context =
                    new SigmaStepContext(0, 1.0d, 0.5d, PredictionType.FLOW);

            assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(sample, modelOutput, context, converter));
            assertArrayEquals(new float[] {1f, 2f}, sample.toFloatArray(), 0f);
        }
    }

    @Test
    void rejectsDerivativeWithShapeDifferentFromSample() {
        try (CpuBackend backend = new CpuBackend();
                Tensor sample = TensorFactory.of(new float[] {1f, 2f}, backend, 2);
                Tensor modelOutput = TensorFactory.of(new float[] {0f, 0f}, backend, 2)) {
            SigmaPredictionConverter converter = converter(
                    PredictionType.FLOW,
                    (current, output, step) -> TensorFactory.of(new float[] {1f}, backend, 1));
            SigmaStepContext context =
                    new SigmaStepContext(0, 1.0d, 0.5d, PredictionType.FLOW);

            assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(sample, modelOutput, context, converter));
            assertArrayEquals(new float[] {1f, 2f}, sample.toFloatArray(), 0f);
        }
    }

    private static SigmaPredictionConverter converter(
            PredictionType predictionType,
            DerivativeFunction function) {
        return new SigmaPredictionConverter() {
            @Override
            public PredictionType predictionType() {
                return predictionType;
            }

            @Override
            public Tensor toDerivative(
                    Tensor sample, Tensor modelOutput, SigmaStepContext context) {
                return function.convert(sample, modelOutput, context);
            }
        };
    }

    @FunctionalInterface
    private interface DerivativeFunction {
        Tensor convert(Tensor sample, Tensor modelOutput, SigmaStepContext context);
    }
}
