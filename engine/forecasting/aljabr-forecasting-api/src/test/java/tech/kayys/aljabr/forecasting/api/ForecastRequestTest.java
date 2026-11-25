package tech.kayys.aljabr.forecasting.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.timeseries.api.TimeSeries;
import tech.kayys.aljabr.timeseries.api.TimeSeriesGroup;

class ForecastRequestTest {
    @Test
    void acceptsExplicitHorizonAndOrderedQuantiles() {
        var series = new TimeSeries("sales", new long[] {1, 2, 3}, new double[] {4, 5, 6});
        var task = new ForecastTask(new TimeSeriesGroup("store-1", List.of(series), List.of()),
                new long[] {4, 5});
        var request = new ForecastRequest(List.of(task), List.of(0.1, 0.5, 0.9));
        assertEquals(2, request.tasks().getFirst().predictionLength());
    }

    @Test
    void rejectsHorizonThatOverlapsHistory() {
        var series = new TimeSeries("sales", new long[] {1, 2, 3}, new double[] {4, 5, 6});
        assertThrows(IllegalArgumentException.class, () -> new ForecastTask(
                new TimeSeriesGroup("store-1", List.of(series), List.of()), new long[] {3, 4}));
    }

    @Test
    void rejectsUnorderedQuantiles() {
        assertThrows(IllegalArgumentException.class, () -> new ForecastRequest(
                List.of(new ForecastTask(
                        new TimeSeriesGroup("store-1",
                                List.of(new TimeSeries("sales", new long[] {1}, new double[] {1})), List.of()),
                        new long[] {2})),
                List.of(0.9, 0.5)));
    }

    @Test
    void forecastOutputDefensivelyCopiesQuantileValues() {
        double[] values = {10, 11};
        var forecast = new SeriesForecast("sales", new long[] {4, 5}, Map.of(0.5, values));
        values[0] = -1;
        assertEquals(10, forecast.quantiles().get(0.5)[0]);
    }
}
