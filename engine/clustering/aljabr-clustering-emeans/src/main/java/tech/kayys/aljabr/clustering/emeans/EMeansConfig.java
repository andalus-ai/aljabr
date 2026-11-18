package tech.kayys.aljabr.clustering.emeans;

import tech.kayys.aljabr.clustering.api.ClusteringConfig;
import tech.kayys.aljabr.clustering.api.DistanceMetric;

/**
 * Explicit CPU e-means training parameters.
 *
 * <p>The batch size and initial step are explicit rather than inferred from
 * the paper's empirical scaling laws. This keeps the first implementation
 * predictable and makes benchmark-driven defaults a separate concern.</p>
 */
public record EMeansConfig(
        int clusterCount,
        long seed,
        int epochs,
        int batchSize,
        double initialStepSize,
        boolean reviveDeadClusters) implements ClusteringConfig {

    public EMeansConfig {
        if (clusterCount < 1) {
            throw new IllegalArgumentException("clusterCount must be positive");
        }
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be positive");
        }
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        if (!Double.isFinite(initialStepSize)
                || initialStepSize <= 0.0 || initialStepSize > 1.0) {
            throw new IllegalArgumentException("initialStepSize must be in (0, 1]");
        }
    }

    @Override
    public DistanceMetric distanceMetric() {
        return DistanceMetric.SQUARED_EUCLIDEAN;
    }

    public static EMeansConfig defaults(int clusterCount, long seed, int epochs, int batchSize) {
        return new EMeansConfig(clusterCount, seed, epochs, batchSize, 0.5, true);
    }
}
