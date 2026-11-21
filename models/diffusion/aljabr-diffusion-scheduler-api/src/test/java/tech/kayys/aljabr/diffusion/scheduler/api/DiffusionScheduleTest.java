package tech.kayys.aljabr.diffusion.scheduler.api;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionScheduleTest {

    @Test
    void acceptsDiscreteTimestepSchedulesWithoutSigma() {
        DiffusionSchedule schedule = new DiffusionSchedule(
                ScheduleDomain.DISCRETE_TIMESTEPS,
                PredictionType.EPSILON,
                List.of(
                        new DiffusionSchedulePoint(0, 999.0, Double.NaN),
                        new DiffusionSchedulePoint(1, 0.0, Double.NaN)));

        assertEquals(2, schedule.points().size());
        assertFalse(schedule.points().getFirst().hasSigma());
    }

    @Test
    void acceptsContinuousSigmaSchedules() {
        DiffusionSchedule schedule = new DiffusionSchedule(
                ScheduleDomain.CONTINUOUS_SIGMA,
                PredictionType.FLOW,
                List.of(new DiffusionSchedulePoint(0, 1.0, 14.0)));

        assertTrue(schedule.points().getFirst().hasSigma());
    }

    @Test
    void rejectsMismatchedSigmaCoordinatesAndNonContiguousIndexes() {
        assertThrows(IllegalArgumentException.class, () -> new DiffusionSchedule(
                ScheduleDomain.CONTINUOUS_SIGMA,
                PredictionType.EPSILON,
                List.of(new DiffusionSchedulePoint(0, 1.0, Double.NaN))));

        assertThrows(IllegalArgumentException.class, () -> new DiffusionSchedule(
                ScheduleDomain.DISCRETE_TIMESTEPS,
                PredictionType.EPSILON,
                List.of(new DiffusionSchedulePoint(1, 1.0, Double.NaN))));
    }

    @Test
    void schedulerDescriptorDeclaresPredictionCompatibility() {
        SchedulerDescriptor descriptor = new SchedulerDescriptor(
                "example-flow",
                ScheduleDomain.CONTINUOUS_FLOW_TIME,
                Set.of(PredictionType.FLOW, PredictionType.VELOCITY),
                false);

        assertTrue(descriptor.supports(PredictionType.FLOW));
        assertFalse(descriptor.supports(PredictionType.EPSILON));
    }
}
