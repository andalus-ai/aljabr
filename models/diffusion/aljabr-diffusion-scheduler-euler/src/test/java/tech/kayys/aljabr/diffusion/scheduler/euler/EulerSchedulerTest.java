package tech.kayys.aljabr.diffusion.scheduler.euler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;
import tech.kayys.aljabr.diffusion.scheduler.api.ScheduleDomain;

class EulerSchedulerTest {

    @Test
    void declaresContinuousSigmaDomainAndSupportedPredictionTypes() {
        EulerScheduler scheduler = new EulerScheduler(0.002d, 80.0d);

        assertEquals("euler", scheduler.descriptor().id());
        assertEquals(ScheduleDomain.CONTINUOUS_SIGMA, scheduler.descriptor().domain());
        assertTrue(scheduler.descriptor().supports(PredictionType.EPSILON));
        assertTrue(scheduler.descriptor().supports(PredictionType.VELOCITY));
        assertTrue(scheduler.descriptor().supports(PredictionType.SAMPLE));
        assertTrue(!scheduler.descriptor().supportsStochasticSampling());
    }

    @Test
    void createsDescendingKarrasSigmaCoordinates() {
        EulerInferenceSchedule schedule =
                new EulerScheduler(0.01d, 100.0d, 7.0d).createInferenceSchedule(5);

        assertEquals(5, schedule.inferenceSteps());
        assertEquals(100.0d, schedule.sigmaAt(0), 1.0e-10d);
        assertEquals(0.01d, schedule.sigmaAt(4), 1.0e-10d);
        for (int i = 1; i < schedule.inferenceSteps(); i++) {
            assertTrue(schedule.sigmaAt(i) < schedule.sigmaAt(i - 1));
        }
        assertEquals(0.0d, schedule.terminalSigma(), 0.0d);
    }

    @Test
    void oneStepUsesMaximumSigmaAndTransitionsToZero() {
        EulerInferenceSchedule schedule =
                new EulerScheduler(0.01d, 10.0d).createInferenceSchedule(1);

        assertEquals(10.0d, schedule.sigmaAt(0), 0.0d);
        assertEquals(0.0d, schedule.nextSigmaAt(0), 0.0d);
    }

    @Test
    void exposesFirstOrderEulerDeltaSigma() {
        EulerScheduler scheduler = new EulerScheduler(0.01d, 10.0d);
        EulerInferenceSchedule schedule = scheduler.createInferenceSchedule(3);

        EulerStepParameters first = scheduler.stepParameters(schedule, 0);
        EulerStepParameters last = scheduler.stepParameters(schedule, 2);

        assertEquals(schedule.sigmaAt(0), first.sigma(), 0.0d);
        assertEquals(schedule.sigmaAt(1), first.nextSigma(), 0.0d);
        assertEquals(first.nextSigma() - first.sigma(), first.deltaSigma(), 0.0d);
        assertEquals(0.0d, last.nextSigma(), 0.0d);
        assertTrue(last.deltaSigma() < 0.0d);
    }

    @Test
    void rejectsInvalidConfigurationAndStepArguments() {
        assertThrows(IllegalArgumentException.class, () -> new EulerScheduler(0.0d, 1.0d));
        assertThrows(IllegalArgumentException.class, () -> new EulerScheduler(1.0d, 1.0d));
        assertThrows(IllegalArgumentException.class, () -> new EulerScheduler(0.1d, 1.0d, 0.0d));
        assertThrows(IllegalArgumentException.class,
                () -> new EulerScheduler(0.1d, 1.0d).createInferenceSchedule(0));
        assertThrows(IllegalArgumentException.class,
                () -> new EulerScheduler(0.1d, 1.0d)
                        .createInferenceSchedule(5, PredictionType.FLOW));

        EulerScheduler scheduler = new EulerScheduler(0.1d, 1.0d);
        EulerInferenceSchedule schedule = scheduler.createInferenceSchedule(3);
        assertThrows(IllegalArgumentException.class, () -> scheduler.stepParameters(schedule, -1));
        assertThrows(IllegalArgumentException.class, () -> scheduler.stepParameters(schedule, 3));
    }
}
