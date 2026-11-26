package tech.kayys.aljabr.diffusion.execution.loop;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.diffusion.execution.api.SigmaPredictionConverter;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepExecutor;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;

class DiffusionDenoisingLoopTest {

    @Test
    void runsOrderedStepsAndReportsProgressWithoutClosingBorrowedInput() {
        try (CpuBackend backend = new CpuBackend();
                Tensor initial = TensorFactory.of(new float[] {1.0f}, backend, 1)) {
            DiffusionDenoisingLoop loop = new DiffusionDenoisingLoop(testExecutor());
            AtomicInteger calls = new AtomicInteger();

            try (Tensor result = loop.run(
                    initial,
                    steps(2),
                    (sample, context) -> {
                        calls.incrementAndGet();
                        return TensorFactory.of(new float[] {1.0f}, backend, 1);
                    },
                    epsilonConverter(backend),
                    () -> false,
                    completed -> assertEquals(calls.get(), completed))) {
                assertEquals(2, calls.get());
                assertArrayEquals(new float[] {-1.0f}, result.toFloatArray(), 0.0001f);
            }
            assertArrayEquals(new float[] {1.0f}, initial.toFloatArray(), 0.0f);
        }
    }

    @Test
    void cancellationClosesLoopOwnedIntermediateButNotInitialInput() {
        try (CpuBackend backend = new CpuBackend();
                Tensor initial = TensorFactory.of(new float[] {1.0f}, backend, 1)) {
            DiffusionDenoisingLoop loop = new DiffusionDenoisingLoop(testExecutor());
            AtomicInteger checks = new AtomicInteger();
            assertThrows(CancellationException.class, () -> loop.run(
                    initial,
                    steps(3),
                    (sample, context) -> TensorFactory.of(new float[] {0.0f}, backend, 1),
                    epsilonConverter(backend),
                    () -> checks.incrementAndGet() > 1,
                    null));
            assertArrayEquals(new float[] {1.0f}, initial.toFloatArray(), 0.0f);
        }
    }

    @Test
    void rejectsDenoiserAliasingBorrowedSampleWithoutClosingIt() {
        try (CpuBackend backend = new CpuBackend();
                Tensor initial = TensorFactory.of(new float[] {1.0f}, backend, 1)) {
            DiffusionDenoisingLoop loop = new DiffusionDenoisingLoop(testExecutor());
            assertThrows(IllegalArgumentException.class, () -> loop.run(
                    initial,
                    steps(1),
                    (sample, context) -> sample,
                    epsilonConverter(backend),
                    null,
                    null));
            assertArrayEquals(new float[] {1.0f}, initial.toFloatArray(), 0.0f);
        }
    }

    @Test
    void rejectsNonContiguousStepIndexesBeforeExecutingModel() {
        try (CpuBackend backend = new CpuBackend();
                Tensor initial = TensorFactory.of(new float[] {1.0f}, backend, 1)) {
            DiffusionDenoisingLoop loop = new DiffusionDenoisingLoop(testExecutor());
            AtomicInteger calls = new AtomicInteger();
            assertThrows(IllegalArgumentException.class, () -> loop.run(
                    initial,
                    List.of(new SigmaStepContext(
                            1, 1.0d, 0.5d, PredictionType.EPSILON)),
                    (sample, context) -> {
                        calls.incrementAndGet();
                        return TensorFactory.of(new float[] {0.0f}, backend, 1);
                    },
                    epsilonConverter(backend),
                    null,
                    null));
            assertEquals(0, calls.get());
            assertArrayEquals(new float[] {1.0f}, initial.toFloatArray(), 0.0f);
        }
    }

    private static List<SigmaStepContext> steps(int count) {
        java.util.ArrayList<SigmaStepContext> contexts = new java.util.ArrayList<>();
        for (int i = 0; i < count; i++) {
            double sigma = count - i;
            double next = i == count - 1 ? 0.0d : sigma - 1.0d;
            contexts.add(new SigmaStepContext(i, sigma, next, PredictionType.EPSILON));
        }
        return List.copyOf(contexts);
    }

    private static SigmaPredictionConverter epsilonConverter(CpuBackend backend) {
        return new SigmaPredictionConverter() {
            @Override
            public PredictionType predictionType() {
                return PredictionType.EPSILON;
            }

            @Override
            public Tensor toDerivative(
                    Tensor sample, Tensor modelOutput, SigmaStepContext context) {
                return TensorFactory.of(modelOutput.toFloatArray(), backend, modelOutput.shape().dims());
            }
        };
    }

    private static SigmaStepExecutor testExecutor() {
        return (sample, modelOutput, context, converter) -> {
            Tensor derivative = converter.toDerivative(sample, modelOutput, context);
            try (derivative; Tensor scaled = derivative.mul((float) context.deltaSigma())) {
                return sample.add(scaled);
            }
        };
    }
}
