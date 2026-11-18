package tech.kayys.aljabr.clustering.api;

/**
 * Common immutable configuration contract shared by clustering algorithms.
 *
 * <p>Algorithm-specific options belong in dedicated configuration types rather
 * than being accumulated in this interface.</p>
 */
public interface ClusteringConfig {
    int clusterCount();

    long seed();

    DistanceMetric distanceMetric();
}
