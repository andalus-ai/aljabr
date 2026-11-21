package tech.kayys.aljabr.models.diffusion;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DDIMSchedulerTest {

    @Test
    void createsDescendingInferenceTimestepsIncludingTrainingEndpoints() {
        DDIMScheduler scheduler = new DDIMScheduler(1000);
        DiffusionSchedule schedule = scheduler.createSchedule(4);

        assertArrayEquals(new int[] {999, 666, 333, 0}, schedule.timesteps());
        assertEquals(1000, schedule.trainingTimesteps());
        assertEquals(4, schedule.inferenceSteps());
    }

    @Test
    void singleInferenceStepStartsAtNoisiestTrainingTimestep() {
        DiffusionSchedule schedule = new DDIMScheduler(20).createSchedule(1);

        assertArrayEquals(new int[] {19}, schedule.timesteps());
        assertEquals(-1, schedule.previousTimestepAt(0));
    }

    @Test
    void etaZeroProducesDeterministicDdimCoefficients() {
        DDIMScheduler scheduler = new DDIMScheduler(100);
        DiffusionSchedule schedule = scheduler.createSchedule(10);

        for (int i = 0; i < schedule.inferenceSteps(); i++) {
            SchedulerStepParameters parameters =
                    scheduler.stepParameters(schedule, i, 0.0f);
            assertEquals(0.0f, parameters.sigma(), 0.0f);
            assertTrue(parameters.directionCoefficient() >= 0.0f);
            assertTrue(parameters.alphaCumprod() > 0.0f);
            assertTrue(parameters.previousAlphaCumprod() > 0.0f);
        }
    }

    @Test
    void etaControlsTheAmountOfStochasticNoise() {
        DDIMScheduler scheduler = new DDIMScheduler(100);
        DiffusionSchedule schedule = scheduler.createSchedule(10);
        SchedulerStepParameters deterministic =
                scheduler.stepParameters(schedule, 0, 0.0f);
        SchedulerStepParameters stochastic =
                scheduler.stepParameters(schedule, 0, 0.5f);

        assertEquals(0.0f, deterministic.sigma(), 0.0f);
        assertTrue(stochastic.sigma() > 0.0f);
        assertTrue(stochastic.sigma() < 1.0f);
    }

    @Test
    void terminalStepUsesCleanSampleAsPreviousAlpha() {
        DDIMScheduler scheduler = new DDIMScheduler(50);
        DiffusionSchedule schedule = scheduler.createSchedule(5);
        SchedulerStepParameters last =
                scheduler.stepParameters(schedule, 4, 0.0f);

        assertEquals(0.0f, last.timestep());
        assertEquals(-1, last.previousTimestep());
        assertEquals(1.0f, last.previousAlphaCumprod(), 0.0f);
        assertEquals(0.0f, last.directionCoefficient(), 1.0e-6f);
    }

    @Test
    void rejectsInvalidConfigurationAndStepArguments() {
        assertThrows(IllegalArgumentException.class, () -> new DDIMScheduler(1));
        assertThrows(IllegalArgumentException.class,
                () -> new DDIMScheduler(10, 0.0f, 0.02f));
        assertThrows(IllegalArgumentException.class,
                () -> new DDIMScheduler(10, 0.03f, 0.02f));

        DDIMScheduler scheduler = new DDIMScheduler(10);
        DiffusionSchedule schedule = scheduler.createSchedule(5);
        assertThrows(IllegalArgumentException.class, () -> scheduler.createSchedule(0));
        assertThrows(IllegalArgumentException.class, () -> scheduler.createSchedule(11));
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.stepParameters(schedule, -1, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.stepParameters(schedule, 0, -0.1f));
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.stepParameters(new DDIMScheduler(20).createSchedule(5), 0, 0.0f));
    }
}
