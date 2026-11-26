package tech.kayys.aljabr.diffusion.execution.euler;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.diffusion.execution.api.SigmaPredictionConverter;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;
import tech.kayys.aljabr.diffusion.scheduler.euler.EulerScheduler;

class EulerDenoisingRunnerTest {

    @Test
    void createsEulerContextsIncludingTerminalZeroSigmaAndRunsAllSteps() {
        try (CpuBackend backend = new CpuBackend();
                Tensor initial = TensorFactory.of(new float[] {10.0f}, backend, 1)) {
            EulerDenoisingRunner runner = new EulerDenoisingRunner(
                    new EulerScheduler(0.5d, 2.0d),
                    new EulerSigmaStepExecutor());
            AtomicInteger completed = new AtomicInteger();

            try (Tensor result = runner.run(
                    initial,
                    2,
                    PredictionType.EPSILON,
                    (sample, context) -> TensorFactory.of(
                            new float[] {1.0f}, backend, 1),
                    converter(backend, PredictionType.EPSILON),
                    () -> false,
                    completed::set)) {
                assertArrayEquals(new float[] {8.0f}, result.toFloatArray(), 0.0001f);
            }

            assertEquals(2, completed.get());
            assertArrayEquals(new float[] {10.0f}, initial.toFloatArray(), 0.0f);
        }
    }

    @Test
    void mapsScheduleToContextsWithCorrectTerminalTransition() {
        var schedule = new EulerScheduler(0.1d, 10.0d)
                .createInferenceSchedule(3, PredictionType.EPSILON);
        var contexts = EulerDenoisingRunner.toStepContexts(schedule, PredictionType.EPSILON);

        assertEquals(3, contexts.size());
        assertEquals(0, contexts.get(0).stepIndex());
        assertEquals(schedule.sigmaAt(0), contexts.get(0).sigma(), 0.0d);
        assertEquals(schedule.sigmaAt(1), contexts.get(0).nextSigma(), 0.0d);
        assertEquals(schedule.sigmaAt(2), contexts.get(1).nextSigma(), 0.0d);
        assertEquals(0.0d, contexts.get(2).nextSigma(), 0.0d);
        assertEquals(true, contexts.get(2).terminalStep());
    }

    @Test
    void rejectsConverterPredictionTypeMismatchBeforeModelEvaluation() {
        try (CpuBackend backend = new CpuBackend();
                Tensor initial = TensorFactory.of(new float[] {1.0f}, backend, 1)) {
            EulerDenoisingRunner runner = new EulerDenoisingRunner(
                    new EulerScheduler(0.1d, 1.0d),
                    new EulerSigmaStepExecutor());
            AtomicInteger calls = new AtomicInteger();

            assertThrows(IllegalArgumentException.class, () -> runner.run(
                    initial,
                    2,
                    PredictionType.EPSILON,
                    (sample, context) -> {
                        calls.incrementAndGet();
                        return TensorFactory.of(new float[] {0.0f}, backend, 1);
                    },
                    converter(backend, PredictionType.VELOCITY),
                    null,
                    null));
            assertEquals(0, calls.get());
            assertArrayEquals(new float[] {1.0f}, initial.toFloatArray(), 0.0f);
        }
    }

    private static SigmaPredictionConverter converter(
            CpuBackend backend, PredictionType predictionType) {
        return new SigmaPredictionConverter() {
            @Override
            public PredictionType predictionType() {
                return predictionType;
            }

            @Override
            public Tensor toDerivative(
                    Tensor sample, Tensor modelOutput, SigmaStepContext context) {
                return TensorFactory.of(modelOutput.toFloatArray(), backend, modelOutput.shape().dims());
            }
        };
    }
}
