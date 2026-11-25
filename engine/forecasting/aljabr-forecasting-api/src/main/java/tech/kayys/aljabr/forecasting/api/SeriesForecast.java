package tech.kayys.aljabr.forecasting.api;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Quantile forecasts for one target series, keyed by probability level. */
public final class SeriesForecast {
    private final String targetId;
    private final long[] timestamps;
    private final Map<Double, double[]> quantiles;

    public SeriesForecast(String targetId, long[] timestamps, Map<Double, double[]> quantiles) {
        this.targetId = Objects.requireNonNull(targetId, "targetId");
        if (targetId.isBlank()) throw new IllegalArgumentException("targetId must not be blank");
        Objects.requireNonNull(timestamps, "timestamps");
        if (timestamps.length == 0) throw new IllegalArgumentException("forecast must not be empty");
        this.timestamps = timestamps.clone();
        for (int i = 1; i < timestamps.length; i++) {
            if (timestamps[i] <= timestamps[i - 1]) throw new IllegalArgumentException("timestamps must increase");
        }
        Objects.requireNonNull(quantiles, "quantiles");
        Map<Double, double[]> copy = new LinkedHashMap<>();
        double previous = 0.0;
        for (var entry : quantiles.entrySet()) {
            double level = entry.getKey();
            double[] values = Objects.requireNonNull(entry.getValue(), "quantile values").clone();
            if (!(level > previous && level < 1.0)) throw new IllegalArgumentException("quantiles must be ordered in (0, 1)");
            if (values.length != timestamps.length) throw new IllegalArgumentException("each quantile must match forecast length");
            for (double value : values) if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("forecast values must be finite");
            }
            copy.put(level, values);
            previous = level;
        }
        if (copy.isEmpty()) throw new IllegalArgumentException("at least one quantile forecast is required");
        this.quantiles = Map.copyOf(copy);
    }

    public String targetId() { return targetId; }
    public long[] timestamps() { return timestamps.clone(); }
    public Map<Double, double[]> quantiles() {
        Map<Double, double[]> copy = new LinkedHashMap<>();
        quantiles.forEach((level, values) -> copy.put(level, values.clone()));
        return Map.copyOf(copy);
    }
}
