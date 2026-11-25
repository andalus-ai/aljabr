package tech.kayys.aljabr.forecasting.api;

import java.util.Collections;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.TreeMap;

/** Quantile forecasts for one target series, keyed by probability level. */
public final class SeriesForecast {
    private final String targetId;
    private final long[] timestamps;
    private final NavigableMap<Double, double[]> quantiles;

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
        NavigableMap<Double, double[]> copy = new TreeMap<>();
        for (var entry : quantiles.entrySet()) {
            Double level = Objects.requireNonNull(entry.getKey(), "quantile level");
            double[] values = Objects.requireNonNull(entry.getValue(), "quantile values").clone();
            if (!(level > 0.0 && level < 1.0)) {
                throw new IllegalArgumentException("quantile levels must be in (0, 1)");
            }
            if (values.length != timestamps.length) {
                throw new IllegalArgumentException("each quantile must match forecast length");
            }
            for (double value : values) {
                if (!Double.isFinite(value)) throw new IllegalArgumentException("forecast values must be finite");
            }
            copy.put(level, values);
        }
        if (copy.isEmpty()) throw new IllegalArgumentException("at least one quantile forecast is required");
        this.quantiles = Collections.unmodifiableNavigableMap(copy);
    }

    public String targetId() { return targetId; }
    public long[] timestamps() { return timestamps.clone(); }

    /** Returns quantile arrays in ascending probability-level order. */
    public Map<Double, double[]> quantiles() {
        NavigableMap<Double, double[]> copy = new TreeMap<>();
        quantiles.forEach((level, values) -> copy.put(level, values.clone()));
        return Collections.unmodifiableNavigableMap(copy);
    }
}
