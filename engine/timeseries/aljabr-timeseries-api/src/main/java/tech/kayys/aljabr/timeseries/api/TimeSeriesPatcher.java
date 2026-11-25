package tech.kayys.aljabr.timeseries.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Extracts left-aligned-to-boundary sliding patches from a single time series.
 *
 * <p>The input is left-padded with missing values until its length is a
 * multiple of the patch size. Complete windows are then extracted using the
 * configured stride. This matches the common forecasting convention where
 * the final observation remains at the right edge of the padded context.
 */
public final class TimeSeriesPatcher {
    private final int patchSize;
    private final int stride;

    public TimeSeriesPatcher(int patchSize, int stride) {
        if (patchSize < 1 || stride < 1) {
            throw new IllegalArgumentException("patch size and stride must be positive");
        }
        this.patchSize = patchSize;
        this.stride = stride;
    }

    public List<Patch> patch(double[] values) {
        return patch(values, null);
    }

    public List<Patch> patch(double[] values, boolean[] observedMask) {
        Objects.requireNonNull(values, "values");
        if (values.length == 0) {
            throw new IllegalArgumentException("values must not be empty");
        }
        if (observedMask != null && observedMask.length != values.length) {
            throw new IllegalArgumentException("observation mask length must match values");
        }

        int leftPadding = (patchSize - values.length % patchSize) % patchSize;
        int paddedLength = Math.addExact(leftPadding, values.length);
        int patchCount = 1 + (paddedLength - patchSize) / stride;
        List<Patch> patches = new ArrayList<>(patchCount);

        for (int patchIndex = 0; patchIndex < patchCount; patchIndex++) {
            int paddedStart = patchIndex * stride;
            double[] patchValues = new double[patchSize];
            boolean[] patchMask = new boolean[patchSize];
            for (int offset = 0; offset < patchSize; offset++) {
                int paddedIndex = paddedStart + offset;
                int sourceIndex = paddedIndex - leftPadding;
                if (sourceIndex < 0) {
                    patchValues[offset] = Double.NaN;
                    continue;
                }
                double value = values[sourceIndex];
                if (Double.isInfinite(value)) {
                    throw new IllegalArgumentException("infinite values are not supported");
                }
                boolean observed = observedMask == null ? !Double.isNaN(value) : observedMask[sourceIndex];
                if (observed && Double.isNaN(value)) {
                    throw new IllegalArgumentException("an observed value cannot be NaN");
                }
                patchValues[offset] = observed ? value : Double.NaN;
                patchMask[offset] = observed;
            }
            patches.add(new Patch(paddedStart - leftPadding, patchValues, patchMask));
        }
        return List.copyOf(patches);
    }

    public int patchSize() { return patchSize; }
    public int stride() { return stride; }

    /** One immutable patch; startIndex is relative to the unpadded input. */
    public record Patch(int startIndex, double[] values, boolean[] observedMask) {
        public Patch {
            values = Objects.requireNonNull(values, "values").clone();
            observedMask = Objects.requireNonNull(observedMask, "observedMask").clone();
            if (values.length == 0 || values.length != observedMask.length) {
                throw new IllegalArgumentException("patch values and mask must have equal non-zero length");
            }
        }

        @Override public double[] values() { return values.clone(); }
        @Override public boolean[] observedMask() { return observedMask.clone(); }

        public boolean hasObservedValues() {
            for (boolean observed : observedMask) {
                if (observed) return true;
            }
            return false;
        }
    }
}
