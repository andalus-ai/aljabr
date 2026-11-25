package tech.kayys.aljabr.forecasting.api;

import java.util.List;
import java.util.Objects;

/** Immutable batch of forecasting tasks and requested probability levels. */
public record ForecastRequest(List<ForecastTask> tasks, List<Double> quantileLevels) {
    public ForecastRequest {
        tasks = List.copyOf(Objects.requireNonNull(tasks, "tasks"));
        quantileLevels = List.copyOf(Objects.requireNonNull(quantileLevels, "quantileLevels"));
        if (tasks.isEmpty()) throw new IllegalArgumentException("at least one forecast task is required");
        if (quantileLevels.isEmpty()) throw new IllegalArgumentException("at least one quantile is required");
        double previous = 0.0;
        for (double level : quantileLevels) {
            if (!(level > previous && level < 1.0)) {
                throw new IllegalArgumentException("quantile levels must be strictly increasing in (0, 1)");
            }
            previous = level;
        }
    }
}
