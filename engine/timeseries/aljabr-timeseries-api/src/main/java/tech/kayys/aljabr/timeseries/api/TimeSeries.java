package tech.kayys.aljabr.timeseries.api;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable, timestamped numeric observations for one target series.
 *
 * <p>Timestamps are epoch milliseconds (or another consistently chosen
 * application time unit); the API does not infer frequency or timezone.
 * Values and timestamps are defensively copied. A false observation-mask
 * entry denotes a missing value; NaN values are also treated as missing.
 */
public final class TimeSeries {
    private final String id;
    private final long[] timestamps;
    private final double[] values;
    private final boolean[] observed;

    public TimeSeries(String id, long[] timestamps, double[] values) {
        this(id, timestamps, values, null);
    }

    public TimeSeries(String id, long[] timestamps, double[] values, boolean[] observed) {
        this.id = requireText(id, "id");
        Objects.requireNonNull(timestamps, "timestamps");
        Objects.requireNonNull(values, "values");
        if (timestamps.length == 0 || timestamps.length != values.length) {
            throw new IllegalArgumentException("timestamps and values must have the same non-zero length");
        }
        if (observed != null && observed.length != values.length) {
            throw new IllegalArgumentException("observed mask length must match values length");
        }
        this.timestamps = timestamps.clone();
        this.values = values.clone();
        this.observed = observed == null ? new boolean[values.length] : observed.clone();
        for (int i = 0; i < timestamps.length; i++) {
            if (i > 0 && timestamps[i] <= timestamps[i - 1]) {
                throw new IllegalArgumentException("timestamps must be strictly increasing");
            }
            if (Double.isInfinite(values[i])) {
                throw new IllegalArgumentException("infinite observations are not supported");
            }
            if (observed == null) {
                this.observed[i] = !Double.isNaN(values[i]);
            } else if (this.observed[i] && Double.isNaN(values[i])) {
                throw new IllegalArgumentException("an observed value cannot be NaN");
            }
        }
    }

    public String id() { return id; }
    public int length() { return values.length; }
    public long timestamp(int index) { return timestamps[index]; }
    public double value(int index) { return values[index]; }
    public boolean isObserved(int index) { return observed[index]; }
    public long[] timestamps() { return timestamps.clone(); }
    public double[] values() { return values.clone(); }
    public boolean[] observedMask() { return observed.clone(); }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }

    @Override
    public String toString() {
        return "TimeSeries[id=" + id + ", length=" + values.length + "]";
    }
}
