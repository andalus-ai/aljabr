package tech.kayys.aljabr.timeseries.api;

import java.util.Objects;

/**
 * Per-series normalization with optional inverse-hyperbolic-sine compression.
 *
 * <p>Statistics are computed from observed values only, using population
 * variance. Missing values remain NaN in the normalized output. The returned
 * state must be retained and used to invert model outputs into source units.
 *
 * <p>This contract is model-neutral. Model families may choose their own
 * normalization settings while sharing the same numerical implementation.
 */
public final class InstanceNormalizer {
    private final double epsilon;
    private final boolean arcsinh;

    public InstanceNormalizer() {
        this(1e-5, false);
    }

    public InstanceNormalizer(double epsilon, boolean arcsinh) {
        if (!(epsilon > 0.0) || !Double.isFinite(epsilon)) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        this.epsilon = epsilon;
        this.arcsinh = arcsinh;
    }

    public Result normalize(double[] values) {
        return normalize(values, null);
    }

    public Result normalize(double[] values, boolean[] observedMask) {
        Objects.requireNonNull(values, "values");
        if (values.length == 0) {
            throw new IllegalArgumentException("values must not be empty");
        }
        if (observedMask != null && observedMask.length != values.length) {
            throw new IllegalArgumentException("observation mask length must match values");
        }

        int count = 0;
        double sum = 0.0;
        boolean[] observed = new boolean[values.length];
        for (int i = 0; i < values.length; i++) {
            double value = values[i];
            if (Double.isInfinite(value)) {
                throw new IllegalArgumentException("infinite values are not supported");
            }
            boolean present = observedMask == null ? !Double.isNaN(value) : observedMask[i];
            if (present && Double.isNaN(value)) {
                throw new IllegalArgumentException("an observed value cannot be NaN");
            }
            observed[i] = present;
            if (present) {
                sum += value;
                count++;
            }
        }

        double location = count == 0 ? 0.0 : sum / count;
        double squaredDeviation = 0.0;
        if (count > 0) {
            for (int i = 0; i < values.length; i++) {
                if (observed[i]) {
                    double delta = values[i] - location;
                    squaredDeviation += delta * delta;
                }
            }
        }
        double scale = count == 0 ? 1.0 : Math.sqrt(squaredDeviation / count);
        if (scale == 0.0) {
            scale = epsilon;
        }

        double[] normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            if (!observed[i]) {
                normalized[i] = Double.NaN;
            } else {
                double standardized = (values[i] - location) / scale;
                normalized[i] = arcsinh ? asinh(standardized) : standardized;
            }
        }
        return new Result(normalized, observed, new State(location, scale, arcsinh));
    }

    private static double asinh(double value) {
        // Stable for large magnitudes and avoids overflow in value * value.
        double magnitude = Math.abs(value);
        double result = Math.log(magnitude + Math.hypot(magnitude, 1.0));
        return Math.copySign(result, value);
    }

    public record Result(double[] values, boolean[] observedMask, State state) {
        public Result {
            values = Objects.requireNonNull(values, "values").clone();
            observedMask = Objects.requireNonNull(observedMask, "observedMask").clone();
            Objects.requireNonNull(state, "state");
            if (values.length == 0 || values.length != observedMask.length) {
                throw new IllegalArgumentException("values and mask must have equal non-zero length");
            }
        }

        @Override public double[] values() { return values.clone(); }
        @Override public boolean[] observedMask() { return observedMask.clone(); }
    }

    /** Immutable statistics and transform settings for inverse normalization. */
    public record State(double location, double scale, boolean arcsinh) {
        public State {
            if (!Double.isFinite(location) || !(scale > 0.0) || !Double.isFinite(scale)) {
                throw new IllegalArgumentException("normalization location and scale must be finite; scale positive");
            }
        }

        public double[] inverse(double[] normalizedValues) {
            Objects.requireNonNull(normalizedValues, "normalizedValues");
            double[] restored = normalizedValues.clone();
            for (int i = 0; i < restored.length; i++) {
                if (!Double.isNaN(restored[i])) {
                    double value = arcsinh ? Math.sinh(restored[i]) : restored[i];
                    restored[i] = value * scale + location;
                }
            }
            return restored;
        }
    }
}
