package tech.kayys.aljabr.clustering.emeans;

import tech.kayys.aljabr.clustering.api.ClusteringConfig;
import tech.kayys.aljabr.clustering.api.DistanceMetric;

/** Explicit CPU e-means parameters. A zero slow-EMA multiplier disables its finalizer. */
public record EMeansConfig(int clusterCount, long seed, int epochs, int batchSize,
        double initialStepSize, boolean reviveDeadClusters, double slowEmaStepMultiplier)
        implements ClusteringConfig {

    /** Backwards-compatible constructor; slow-EMA finalization is disabled. */
    public EMeansConfig(int clusterCount, long seed, int epochs, int batchSize,
            double initialStepSize, boolean reviveDeadClusters) {
        this(clusterCount, seed, epochs, batchSize, initialStepSize, reviveDeadClusters, 0.0);
    }

    public EMeansConfig {
        if (clusterCount < 1) throw new IllegalArgumentException("clusterCount must be positive");
        if (epochs < 1) throw new IllegalArgumentException("epochs must be positive");
        if (batchSize < 1) throw new IllegalArgumentException("batchSize must be positive");
        if (!Double.isFinite(initialStepSize) || initialStepSize <= 0.0 || initialStepSize > 1.0)
            throw new IllegalArgumentException("initialStepSize must be in (0, 1]");
        if (!Double.isFinite(slowEmaStepMultiplier)
                || slowEmaStepMultiplier < 0.0 || slowEmaStepMultiplier > 4.0)
            throw new IllegalArgumentException("slowEmaStepMultiplier must be in [0, 4]");
    }

    @Override public DistanceMetric distanceMetric() { return DistanceMetric.SQUARED_EUCLIDEAN; }

    /** Enables slow-EMA finalization at a multiplier of the final cosine-floor step. */
    public EMeansConfig withSlowEmaStepMultiplier(double multiplier) {
        return new EMeansConfig(clusterCount, seed, epochs, batchSize,
                initialStepSize, reviveDeadClusters, multiplier);
    }

    public static EMeansConfig auto(int clusterCount, long seed, int epochs,
            long sampleCount, int dimensions) {
        EMeansParameterScaling.Recommendation r =
                EMeansParameterScaling.recommend(sampleCount, clusterCount, epochs, dimensions);
        return new EMeansConfig(clusterCount, seed, epochs, r.batchSize(), r.initialStepSize(), true);
    }

    public static EMeansConfig defaults(int clusterCount, long seed, int epochs, int batchSize) {
        return new EMeansConfig(clusterCount, seed, epochs, batchSize, 0.5, true);
    }
}
