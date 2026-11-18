package tech.kayys.aljabr.clustering.emeans;

/**
 * Parameter recommendation from the e-means paper's fitted scaling laws.
 *
 * <p>The fit was trained on high-dimensional embedding data. The paper
 * explicitly cautions that it is not optimal for tiny or very low-dimensional
 * problems, so callers should treat these values as starting recommendations,
 * not universal guarantees.</p>
 *
 * @see <a href="https://arxiv.org/abs/2611.09501">Bessa et al., 2026, Table 2</a>
 */
public final class EMeansParameterScaling {
    private static final double STEP_COEFFICIENT = 0.824;
    private static final double BATCH_COEFFICIENT = 0.035;
    private static final long LARGE_DATASET_THRESHOLD = 2_000_000L;

    private EMeansParameterScaling() {
    }

    /**
     * Recommends batch size and initial EMA step from sample count, cluster
     * count, and epoch budget using the paper's Table 2 power laws.
     *
     * <p>For datasets above two million samples, the fitted batch estimate is
     * divided by eight before power-of-two rounding. The final batch is capped
     * at the sample count.</p>
     */
    public static Recommendation recommend(
            long sampleCount, int clusterCount, int epochs, int dimensions) {
        if (sampleCount < 1) {
            throw new IllegalArgumentException("sampleCount must be positive");
        }
        if (clusterCount < 1) {
            throw new IllegalArgumentException("clusterCount must be positive");
        }
        if (clusterCount > sampleCount) {
            throw new IllegalArgumentException("clusterCount must not exceed sampleCount");
        }
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be positive");
        }
        if (dimensions < 1) {
            throw new IllegalArgumentException("dimensions must be positive");
        }

        double n = sampleCount;
        double k = clusterCount;
        double t = epochs;
        double rawBatch = BATCH_COEFFICIENT
                * Math.pow(k, 0.13)
                * Math.pow(n, 0.88)
                * Math.pow(t, 0.81);
        if (sampleCount > LARGE_DATASET_THRESHOLD) {
            rawBatch /= 8.0;
        }
        long roundedBatch = nearestPowerOfTwo(rawBatch);
        int batchSize = (int) Math.min(Math.min(sampleCount, Integer.MAX_VALUE), roundedBatch);

        double alpha = STEP_COEFFICIENT
                * Math.pow(k, -0.46)
                * Math.pow(batchSize, 0.90)
                * Math.pow(n, -0.45)
                * Math.pow(t, -0.11);
        alpha = Math.min(alpha, 1.0);

        return new Recommendation(
                batchSize,
                alpha,
                dimensions <= 16,
                sampleCount > LARGE_DATASET_THRESHOLD);
    }

    private static long nearestPowerOfTwo(double value) {
        if (!Double.isFinite(value) || value >= (1L << 62)) {
            return 1L << 62;
        }
        if (value <= 1.0) {
            return 1L;
        }
        long lower = Long.highestOneBit((long) Math.floor(value));
        long upper = lower << 1;
        return value - lower < upper - value ? lower : upper;
    }

    /** Immutable result with an explicit low-dimensional caution. */
    public record Recommendation(
            int batchSize,
            double initialStepSize,
            boolean lowDimensionalCaution,
            boolean largeDatasetAdjustment) {
    }
}
