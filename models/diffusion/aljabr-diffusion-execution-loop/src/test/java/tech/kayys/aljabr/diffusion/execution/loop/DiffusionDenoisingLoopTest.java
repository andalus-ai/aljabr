package tech.kayys.aljabr.diffusion.execution.loop;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.execution.api.SigmaPredictionConverter;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.execution.euler.EulerSigmaStepExecutor;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;

class DiffusionDenoisingLoopTest {

    private final DiffusionDenoisingLoop loop =
            new DiffusionDenoisingLoop(new EulerSigmaStepExecutor());

    @Test
    void runsOrderedStepsAndReportsProgressWithoutClosingBorrowedInput() {
        try (Tensor initial = Tensor.of(new float[]{1.0f}, 1)) {
            AtomicInteger calls = new AtomicInteger();
            try (Tensor result = loop.run(
                    initial,
                    steps(2),
                    (sample, context) -> {
                        calls.incrementAndGet();
                        return Tensor.of(new float[]{1.0f}, 1);
                    },
                    epsilonConverter(),
                    () -> false,
                    completed -> assertEquals(calls.get(), completed))) {
                assertEquals(2, calls.get());
                assertTrue(initial.item() == 1.0f);
                assertTrue(result.item() < 1.0f);
            }
            assertEquals(1.0f, initial.get(0));
        }
    }

    @Test
    void cancellationClosesLoopOwnedIntermediateButNotInitialInput() {
        try (Tensor initial = Tensor.of(new float[]{1.0f}, 1)) {
            AtomicInteger checks = new AtomicInteger();
            assertThrows(CancellationException.class, () -> loop.run(
                    initial,
                    steps(3),
                    (sample, context) -> Tensor.of(new float[]{0.0f}, 1),
                    epsilonConverter(),
                    () -> checks.incrementAndGet() > 1,
                    null));
            assertEquals(1.0f, initial.get(0));
        }
    }

    @Test
    void rejectsDenoiserAliasingBorrowedSampleWithoutClosingIt() {
        try (Tensor initial = Tensor.of(new float[]{1.0f}, 1)) {
            assertThrows(IllegalArgumentException.class, () -> loop.run(
                    initial,
                    steps(1),
                    (sample, context) -> sample,
                    epsilonConverter(),
                    null,
                    null));
            assertEquals(1.0f, initial.get(0));
        }
    }

    @Test
    void rejectsNonContiguousStepIndexesBeforeExecutingModel() {
        try (Tensor initial = Tensor.of(new float[]{1.0f}, 1)) {
            AtomicInteger calls = new AtomicInteger();
            assertThrows(IllegalArgumentException.class, () -> loop.run(
                    initial,
                    List.of(new SigmaStepContext(
                            1, 1.0d, 0.5d, PredictionType.EPSILON)),
                    (sample, context) -> {
                        calls.incrementAndGet();
                        return Tensor.of(new float[]{0.0f}, 1);
                    },
                    epsilonConverter(),
                    null,
                    null));
            assertEquals(0, calls.get());
            assertEquals(1.0f, initial.get(0));
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

    private static SigmaPredictionConverter epsilonConverter() {
        return new SigmaPredictionConverter() {
            @Override
            public PredictionType predictionType() {
                return PredictionType.EPSILON;
            }

            @Override
            public Tensor toDerivative(
                    Tensor sample, Tensor modelOutput, SigmaStepContext context) {
                // For this test fixture, modelOutput is already the derivative.
                return Tensor.of(new float[]{modelOutput.get(0)}, 1);
            }
        };
    }
}
