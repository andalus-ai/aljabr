package tech.kayys.aljabr.timeseries.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TimeSeriesTest {
    @Test
    void infersMissingObservationsFromNanAndDefensivelyCopiesArrays() {
        long[] timestamps = {1, 2, 3};
        double[] values = {2.0, Double.NaN, 4.0};
        var series = new TimeSeries("temperature", timestamps, values);
        timestamps[0] = 99;
        values[0] = 99;
        assertEquals(1, series.timestamp(0));
        assertEquals(2.0, series.value(0));
        assertFalse(series.isObserved(1));
    }

    @Test
    void rejectsNonIncreasingTimestamps() {
        assertThrows(IllegalArgumentException.class,
                () -> new TimeSeries("temperature", new long[] {1, 1}, new double[] {2, 3}));
    }

    @Test
    void supportsCategoricalKnownFutureCovariates() {
        var covariate = CovariateSeries.categorical(
                "promotion", new long[] {1, 2, 3}, new String[] {"none", null, "sale"}, true);
        assertEquals(CovariateSeries.Type.CATEGORICAL, covariate.type());
        assertEquals("sale", covariate.categoricalValues()[2]);
        assertFalse(covariate.isObserved(1));
    }
}
