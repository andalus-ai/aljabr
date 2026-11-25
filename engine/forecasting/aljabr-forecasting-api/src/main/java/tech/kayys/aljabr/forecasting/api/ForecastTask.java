package tech.kayys.aljabr.forecasting.api;

import java.util.Objects;
import tech.kayys.aljabr.timeseries.api.TimeSeriesGroup;

/** One isolated group and its explicit forecast timestamps. */
public final class ForecastTask {
    private final TimeSeriesGroup group;
    private final long[] forecastTimestamps;

    public ForecastTask(TimeSeriesGroup group, long[] forecastTimestamps) {
        this.group = Objects.requireNonNull(group, "group");
        Objects.requireNonNull(forecastTimestamps, "forecastTimestamps");
        if (forecastTimestamps.length == 0) throw new IllegalArgumentException("forecast horizon must not be empty");
        this.forecastTimestamps = forecastTimestamps.clone();
        for (int i = 0; i < forecastTimestamps.length; i++) {
            if (i > 0 && forecastTimestamps[i] <= forecastTimestamps[i - 1]) {
                throw new IllegalArgumentException("forecast timestamps must be strictly increasing");
            }
            for (var target : group.targets()) {
                if (forecastTimestamps[i] <= target.timestamp(target.length() - 1)) {
                    throw new IllegalArgumentException("forecast timestamps must follow target history");
                }
            }
        }
        for (var covariate : group.covariates()) {
            if (covariate.knownFuture()) {
                long[] known = covariate.timestamps();
                if (known[known.length - 1] < forecastTimestamps[forecastTimestamps.length - 1]) {
                    throw new IllegalArgumentException("known-future covariate does not cover the forecast horizon: "
                            + covariate.name());
                }
            }
        }
    }

    public TimeSeriesGroup group() { return group; }
    public long[] forecastTimestamps() { return forecastTimestamps.clone(); }
    public int predictionLength() { return forecastTimestamps.length; }
}
