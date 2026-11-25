package tech.kayys.aljabr.forecasting.api;

import java.util.List;
import java.util.Objects;

/** Results for one input group; output target IDs remain explicit. */
public record ForecastResult(List<SeriesForecast> forecasts) {
    public ForecastResult {
        forecasts = List.copyOf(Objects.requireNonNull(forecasts, "forecasts"));
        if (forecasts.isEmpty()) throw new IllegalArgumentException("forecast result must not be empty");
    }
}
