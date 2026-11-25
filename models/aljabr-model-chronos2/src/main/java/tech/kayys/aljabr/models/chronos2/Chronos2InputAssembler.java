package tech.kayys.aljabr.models.chronos2;

import java.util.Arrays;
import java.util.Objects;
import tech.kayys.aljabr.timeseries.api.InstanceNormalizer;
import tech.kayys.aljabr.timeseries.api.TimeSeriesPatcher;

/**
 * Builds the numeric patch features consumed by the Chronos-2 encoder.
 *
 * <p>Each timestep contributes three channels: normalized time index,
 * normalized target/covariate value, and observation mask. This class owns
 * only Chronos-2's feature layout; normalization and patch extraction are
 * shared time-series primitives.
 */
public final class Chronos2InputAssembler {
    private final Chronos2Config config;
    private final InstanceNormalizer normalizer;
    private final TimeSeriesPatcher patcher;

    public Chronos2InputAssembler(Chronos2Config config) {
        this.config = Objects.requireNonNull(config, "config");
        this.normalizer = new InstanceNormalizer(1e-5, config.useArcsinh());
        this.patcher = new TimeSeriesPatcher(config.inputPatchSize(), config.inputPatchStride());
    }

    /**
     * Prepares one target series and its optional known-future real covariate.
     *
     * @param context historical target observations
     * @param contextMask optional observation mask; null infers missing values from NaN
     * @param futureCovariates optional numeric future covariate aligned to forecast steps
     * @param futureCovariatesMask optional known-value mask; null infers from NaN
     * @param numOutputPatches number of future patches requested from the encoder
     */
    public Input assemble(
            double[] context,
            boolean[] contextMask,
            double[] futureCovariates,
            boolean[] futureCovariatesMask,
            int numOutputPatches) {
        Objects.requireNonNull(context, "context");
        if (context.length == 0) {
            throw new IllegalArgumentException("context must not be empty");
        }
        if (contextMask != null && contextMask.length != context.length) {
            throw new IllegalArgumentException("context mask length must match context");
        }
        if (futureCovariates == null && futureCovariatesMask != null) {
            throw new IllegalArgumentException("future covariate mask requires future covariates");
        }
        if (futureCovariates != null && futureCovariatesMask != null
                && futureCovariates.length != futureCovariatesMask.length) {
            throw new IllegalArgumentException("future covariate mask length must match values");
        }
        if (numOutputPatches < 1 || numOutputPatches > config.maxOutputPatches()) {
            throw new IllegalArgumentException("numOutputPatches exceeds configured capacity");
        }

        int contextStart = Math.max(0, context.length - config.contextLength());
        double[] boundedContext = Arrays.copyOfRange(context, contextStart, context.length);
        boolean[] boundedContextMask = contextMask == null
                ? null : Arrays.copyOfRange(contextMask, contextStart, contextMask.length);

        InstanceNormalizer.Result normalized = normalizer.normalize(boundedContext, boundedContextMask);
        InstanceNormalizer.State normalizationState = normalized.state();
        var contextPatches = patcher.patch(normalized.values(), normalized.observedMask());
        int patchSize = config.inputPatchSize();
        int contextPatchCount = contextPatches.size();
        double[][] contextFeatures = new double[contextPatchCount][patchSize * 3];
        boolean[] contextAttentionMask = new boolean[contextPatchCount];

        int finalContextLength = Math.multiplyExact(contextPatchCount, patchSize);
        for (int patchIndex = 0; patchIndex < contextPatchCount; patchIndex++) {
            TimeSeriesPatcher.Patch patch = contextPatches.get(patchIndex);
            double[] values = patch.values();
            boolean[] observed = patch.observedMask();
            contextAttentionMask[patchIndex] = patch.hasObservedValues();
            for (int offset = 0; offset < patchSize; offset++) {
                int feature = offset;
                contextFeatures[patchIndex][feature] =
                        (double) (-finalContextLength + patchIndex * patchSize + offset)
                                / config.timeEncodingScale();
                contextFeatures[patchIndex][patchSize + feature] = observed[offset] ? values[offset] : 0.0;
                contextFeatures[patchIndex][2 * patchSize + feature] = observed[offset] ? 1.0 : 0.0;
            }
        }

        int outputLength = Math.multiplyExact(numOutputPatches, config.outputPatchSize());
        double[] normalizedFuture = new double[outputLength];
        boolean[] futureMask = new boolean[outputLength];
        if (futureCovariates != null) {
            if (futureCovariates.length > outputLength) {
                throw new IllegalArgumentException("future covariates exceed requested output horizon");
            }
            boolean[] suppliedMask = futureCovariatesMask == null
                    ? null : futureCovariatesMask.clone();
            normalizedFuture = normalizationState.transform(futureCovariates, suppliedMask);
            for (int i = 0; i < futureCovariates.length; i++) {
                futureMask[i] = suppliedMask == null
                        ? !Double.isNaN(futureCovariates[i]) : suppliedMask[i];
            }
        }

        double[][] futureFeatures = new double[numOutputPatches][patchSize * 3];
        for (int patchIndex = 0; patchIndex < numOutputPatches; patchIndex++) {
            for (int offset = 0; offset < patchSize; offset++) {
                int index = patchIndex * patchSize + offset;
                futureFeatures[patchIndex][offset] = (double) index / config.timeEncodingScale();
                futureFeatures[patchIndex][patchSize + offset] =
                        futureMask[index] ? normalizedFuture[index] : 0.0;
                futureFeatures[patchIndex][2 * patchSize + offset] =
                        futureMask[index] ? 1.0 : 0.0;
            }
        }

        return new Input(
                contextFeatures,
                contextAttentionMask,
                futureFeatures,
                futureMask,
                normalizationState,
                contextPatchCount,
                numOutputPatches);
    }

    /** Immutable encoder input plus the fitted transform required for output inversion. */
    public record Input(
            double[][] contextFeatures,
            boolean[] contextAttentionMask,
            double[][] futureFeatures,
            boolean[] futureCovariateMask,
            InstanceNormalizer.State normalizationState,
            int contextPatchCount,
            int outputPatchCount) {
        public Input {
            contextFeatures = deepCopy(contextFeatures);
            contextAttentionMask = Objects.requireNonNull(contextAttentionMask, "contextAttentionMask").clone();
            futureFeatures = deepCopy(futureFeatures);
            futureCovariateMask = Objects.requireNonNull(futureCovariateMask, "futureCovariateMask").clone();
            Objects.requireNonNull(normalizationState, "normalizationState");
            if (contextFeatures.length != contextAttentionMask.length
                    || futureFeatures.length == 0
                    || contextPatchCount != contextFeatures.length
                    || outputPatchCount != futureFeatures.length
                    || futureCovariateMask.length != futureFeatures.length * (contextFeatures[0].length / 3)) {
                throw new IllegalArgumentException("encoder input dimensions are inconsistent");
            }
        }

        @Override public double[][] contextFeatures() { return deepCopy(contextFeatures); }
        @Override public boolean[] contextAttentionMask() { return contextAttentionMask.clone(); }
        @Override public double[][] futureFeatures() { return deepCopy(futureFeatures); }
        @Override public boolean[] futureCovariateMask() { return futureCovariateMask.clone(); }

        private static double[][] deepCopy(double[][] values) {
            Objects.requireNonNull(values, "values");
            double[][] copy = new double[values.length][];
            for (int i = 0; i < values.length; i++) {
                copy[i] = Objects.requireNonNull(values[i], "row").clone();
            }
            return copy;
        }
    }
}
