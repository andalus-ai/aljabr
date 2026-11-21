package tech.kayys.aljabr.diffusion.scheduler.ddim;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;
import tech.kayys.aljabr.diffusion.scheduler.api.ScheduleDomain;

class DDIMSchedulerTest {

    @Test
    void declaresDiscreteEpsilonPredictionAndStochasticSampling() {
        DDIMScheduler scheduler = new DDIMScheduler(1000);

        assertEquals("ddim", scheduler.descriptor().id());
        assertEquals(ScheduleDomain.DISCRETE_TIMESTEPS, scheduler.descriptor().domain());
        assertTrue(scheduler.descriptor().supports(PredictionType.EPSILON));
        assertTrue(scheduler.descriptor().supportsStochasticSampling());
    }

    @Test
    void createsDescendingInferenceTimestepsIncludingTrainingEndpoints() {
        DDIMInferenceSchedule schedule = new DDIMScheduler(1000).createInferenceSchedule(4);

        assertArrayEquals(new int[] {999, 666, 333, 0}, schedule.timesteps());
        assertEquals(1000, schedule.trainingTimesteps());
        assertEquals(4, schedule.inferenceSteps());
    }

    @Test
    void singleInferenceStepStartsAtNoisiestTrainingTimestep() {
        DDIMInferenceSchedule schedule = new DDIMScheduler(20).createInferenceSchedule(1);

        assertArrayEquals(new int[] {19}, schedule.timesteps());
        assertEquals(-1, schedule.previousTimestepAt(0));
    }

    @Test
    void etaZeroProducesDeterministicDdimCoefficients() {
        DDIMScheduler scheduler = new DDIMScheduler(100);
        DDIMInferenceSchedule schedule = scheduler.createInferenceSchedule(10);

        for (int i = 0; i < schedule.inferenceSteps(); i++) {
            DDIMStepParameters parameters = scheduler.stepParameters(schedule, i, 0.0f);
            assertEquals(0.0f, parameters.sigma(), 0.0f);
            assertTrue(parameters.directionCoefficient() >= 0.0f);
            assertTrue(parameters.alphaCumprod() > 0.0f);
            assertTrue(parameters.previousAlphaCumprod() > 0.0f);
        }
    }

    @Test
    void etaControlsTheAmountOfStochasticNoise() {
        DDIMScheduler scheduler = new DDIMScheduler(100);
        DDIMInferenceSchedule schedule = scheduler.createInferenceSchedule(10);
        DDIMStepParameters deterministic = scheduler.stepParameters(schedule, 0, 0.0f);
        DDIMStepParameters stochastic = scheduler.stepParameters(schedule, 0, 0.5f);

        assertEquals(0.0f, deterministic.sigma(), 0.0f);
        assertTrue(stochastic.sigma() > 0.0f);
        assertTrue(stochastic.sigma() < 1.0f);
    }

    @Test
    void terminalStepUsesCleanSampleAsPreviousAlpha() {
        DDIMScheduler scheduler = new DDIMScheduler(50);
        DDIMInferenceSchedule schedule = scheduler.createInferenceSchedule(5);
        DDIMStepParameters last = scheduler.stepParameters(schedule, 4, 0.0f);

        assertEquals(0, last.timestep());
        assertEquals(-1, last.previousTimestep());
        assertEquals(1.0f, last.previousAlphaCumprod(), 0.0f);
        assertEquals(0.0f, last.directionCoefficient(), 1.0e-6f);
    }

    @Test
    void rejectsUnsupportedPredictionAndInvalidArguments() {
        DDIMScheduler scheduler = new DDIMScheduler(10);
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.createSchedule(5, PredictionType.FLOW));
        assertThrows(IllegalArgumentException.class, () -> scheduler.createInferenceSchedule(0));
        assertThrows(IllegalArgumentException.class, () -> scheduler.createInferenceSchedule(11));
        assertThrows(IllegalArgumentException.class, () -> new DDIMScheduler(1));
        assertThrows(IllegalArgumentException.class, () -> new DDIMScheduler(10, 0.0f, 0.02f));

        DDIMInferenceSchedule schedule = scheduler.createInferenceSchedule(5);
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.stepParameters(schedule, -1, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.stepParameters(schedule, 0, -0.1f));
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.stepParameters(
                        new DDIMScheduler(20).createInferenceSchedule(5), 0, 0.0f));
    }
}
