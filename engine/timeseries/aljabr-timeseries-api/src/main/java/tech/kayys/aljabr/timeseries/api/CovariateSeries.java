package tech.kayys.aljabr.timeseries.api;

import java.util.Objects;

/**
 * A timestamped feature associated with one or more target series.
 *
 * <p>Real and categorical covariates share one contract. A known-future
 * covariate may contain timestamps beyond the target context; a historical
 * covariate must not be interpreted as available in the forecast horizon.
 * NaN (real) and null (categorical) values are missing unless explicitly
 * marked otherwise by an observation mask.
 */
public final class CovariateSeries {
    public enum Type { REAL, CATEGORICAL }

    private final String name;
    private final Type type;
    private final long[] timestamps;
    private final double[] realValues;
    private final String[] categoricalValues;
    private final boolean[] observed;
    private final boolean knownFuture;

    private CovariateSeries(String name, Type type, long[] timestamps,
            double[] realValues, String[] categoricalValues, boolean[] observed, boolean knownFuture) {
        this.name = requireText(name);
        this.type = Objects.requireNonNull(type, "type");
        Objects.requireNonNull(timestamps, "timestamps");
        if (timestamps.length == 0) throw new IllegalArgumentException("covariate must contain observations");
        this.timestamps = timestamps.clone();
        for (int i = 1; i < timestamps.length; i++) {
            if (timestamps[i] <= timestamps[i - 1]) {
                throw new IllegalArgumentException("covariate timestamps must be strictly increasing");
            }
        }
        this.realValues = realValues == null ? null : realValues.clone();
        this.categoricalValues = categoricalValues == null ? null : categoricalValues.clone();
        int length = timestamps.length;
        if (observed != null && observed.length != length) {
            throw new IllegalArgumentException("observation mask length must match timestamps");
        }
        this.observed = observed == null ? new boolean[length] : observed.clone();
        if (type == Type.REAL) {
            if (realValues == null || realValues.length != length || categoricalValues != null) {
                throw new IllegalArgumentException("real covariates require one numeric value per timestamp");
            }
            for (int i = 0; i < length; i++) {
                double value = realValues[i];
                if (Double.isInfinite(value)) throw new IllegalArgumentException("infinite covariates are not supported");
                if (observed == null) this.observed[i] = !Double.isNaN(value);
                else if (this.observed[i] && Double.isNaN(value)) {
                    throw new IllegalArgumentException("an observed real covariate cannot be NaN");
                }
            }
        } else {
            if (categoricalValues == null || categoricalValues.length != length || realValues != null) {
                throw new IllegalArgumentException("categorical covariates require one category per timestamp");
            }
            for (int i = 0; i < length; i++) {
                if (observed == null) this.observed[i] = categoricalValues[i] != null;
                else if (this.observed[i] && categoricalValues[i] == null) {
                    throw new IllegalArgumentException("an observed categorical covariate cannot be null");
                }
            }
        }
        this.knownFuture = knownFuture;
    }

    public static CovariateSeries real(String name, long[] timestamps, double[] values, boolean knownFuture) {
        return real(name, timestamps, values, null, knownFuture);
    }

    public static CovariateSeries real(
            String name, long[] timestamps, double[] values, boolean[] observed, boolean knownFuture) {
        return new CovariateSeries(name, Type.REAL, timestamps, values, null, observed, knownFuture);
    }

    public static CovariateSeries categorical(
            String name, long[] timestamps, String[] values, boolean knownFuture) {
        return categorical(name, timestamps, values, null, knownFuture);
    }

    public static CovariateSeries categorical(
            String name, long[] timestamps, String[] values, boolean[] observed, boolean knownFuture) {
        return new CovariateSeries(name, Type.CATEGORICAL, timestamps, null, values, observed, knownFuture);
    }

    public String name() { return name; }
    public Type type() { return type; }
    public boolean knownFuture() { return knownFuture; }
    public int length() { return timestamps.length; }
    public long[] timestamps() { return timestamps.clone(); }
    public boolean isObserved(int index) { return observed[index]; }
    public boolean[] observedMask() { return observed.clone(); }

    public double[] realValues() {
        if (type != Type.REAL) throw new IllegalStateException("not a real covariate");
        return realValues.clone();
    }

    public String[] categoricalValues() {
        if (type != Type.CATEGORICAL) throw new IllegalStateException("not a categorical covariate");
        return categoricalValues.clone();
    }

    private static String requireText(String value) {
        Objects.requireNonNull(value, "name");
        if (value.isBlank()) throw new IllegalArgumentException("name must not be blank");
        return value;
    }
}
